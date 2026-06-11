// 1D elementary cellular automaton simulator.
// Supports all 256 Wolfram rules (0-255).
// Bit-packing layout: Vec<u64>, MSB-first. Bit 63 of word[i] = leftmost cell in that word.
// Cell position `pos` maps to: word = pos / 64, bit = 63 - (pos % 64).
// Boundary condition: cells beyond left/right edges are always 0 (dead).
// Initial state: all cells 0 except center cell (index width / 2) set to 1.
// Output: grayscale PNG. Live cell (1) = 0x00 (black). Dead cell (0) = 0xFF (white).
// Memory: only 2 bitpacked row buffers (current + next) plus the flat image buffer.

use clap::Parser;
use image::GrayImage;
use std::io::Write;
use std::time::Instant;

/// CLI arguments. Parsed by clap. All `///` comments here become `--help` text for human users.
#[derive(Parser)]
#[command(name = "automaton")]
struct Args {
    /// Wolfram rule number (0-255)
    #[arg(long)]
    rule: u8,

    /// Number of generations to simulate
    #[arg(long)]
    steps: u32,

    /// Override width in cells, minimum 1 (default: 2 * steps + 1)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    width: Option<u32>,

    /// Output PNG file path
    #[arg(long)]
    output: String,

    /// Skip confirmation prompt for large simulations (image buffer over 200 MB)
    #[arg(long, default_value_t = false)]
    force: bool,
}

/// Image buffers larger than this many bytes (1 byte per pixel) trigger a confirmation
/// prompt unless --force is set. 200 MB is roughly a default-width run at 10_000 steps
/// (20_001 x 10_001 pixels).
const LARGE_BUFFER_THRESHOLD: u64 = 200 * 1024 * 1024;

fn main() {
    let args = Args::parse();

    // Width defaults to 2 * steps + 1 (triangle pattern grows 1 cell per side per generation).
    // Uses checked arithmetic to prevent u32 overflow when steps is very large.
    let width = args.width.unwrap_or_else(|| {
        args.steps.checked_mul(2).and_then(|v| v.checked_add(1)).unwrap_or_else(|| {
            eprintln!("Error: steps too large, width would overflow u32.");
            std::process::exit(1);
        })
    });
    // Height = steps + 1 because row 0 is the initial state, rows 1..=steps are computed generations.
    // Checked because steps == u32::MAX would wrap height to 0.
    let height = args.steps.checked_add(1).unwrap_or_else(|| {
        eprintln!("Error: steps too large, height would overflow u32.");
        std::process::exit(1);
    });

    // Guard against large allocations. Image buffer = width * height bytes (1 byte per pixel),
    // so the guard is on byte count: a huge --width with few steps must trigger it too.
    // At steps=50000 with default width, this is ~5 GB.
    let pixels = width as u64 * height as u64;
    if pixels > LARGE_BUFFER_THRESHOLD && !args.force {
        let megabytes = pixels / (1024 * 1024);
        eprintln!(
            "Warning: {} steps with width {} = {} pixels (~{} MB image buffer).",
            args.steps, width, pixels, megabytes
        );
        eprintln!("This will produce a very large PNG file.");
        eprint!("Continue? [y/N] ");
        std::io::stderr().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("Aborted. Use --force to skip this prompt.");
            std::process::exit(1);
        }
    }

    let start = Instant::now();

    let rule_table = build_rule_table(args.rule);
    // Each u64 word holds 64 cells. Round up to cover all `width` cells.
    let num_words = (width as usize).div_ceil(64);

    // Flat image buffer: width * height bytes, pre-filled with 0xFF (white/dead).
    // Row `r` starts at index r * width. Each byte is one pixel.
    // decode_row_to_image relies on this pre-fill: it writes only live (0x00) pixels.
    let mut img_buf: Vec<u8> = vec![0xFF; width as usize * height as usize];

    // Two bitpacked row buffers. Only these two exist at any time — not all generations.
    // After each generation: swap current/next, zero out next for reuse.
    let mut current = vec![0u64; num_words];
    let mut next = vec![0u64; num_words];

    // Initial state: single live cell at center position.
    let center = width as usize / 2;
    set_bit(&mut current, center);

    // Write initial state (row 0) into image buffer.
    decode_row_to_image(&current, &mut img_buf, 0, width as usize);

    // Main simulation loop. For each generation:
    // 1. Compute next row from current using rule lookup.
    // 2. Write next row's pixels into image buffer.
    // 3. Swap buffers and zero the old one.
    for row_idx in 1..height as usize {
        compute_next_generation(&current, &mut next, &rule_table, width as usize);
        decode_row_to_image(&next, &mut img_buf, row_idx, width as usize);
        std::mem::swap(&mut current, &mut next);
        next.fill(0);
    }

    // Write the flat pixel buffer as a grayscale PNG (8-bit, 1 channel).
    let img = GrayImage::from_raw(width, height, img_buf)
        .expect("Failed to create image from buffer");
    img.save(&args.output).unwrap_or_else(|e| {
        eprintln!("Error: failed to save PNG to {}: {}", args.output, e);
        std::process::exit(1);
    });

    let elapsed = start.elapsed();
    eprintln!(
        "Rule {} | {} steps | {}x{} | {:.3}s | saved to {}",
        args.rule, args.steps, width, height, elapsed.as_secs_f64(), args.output
    );
}

/// Converts a Wolfram rule number into an 8-entry lookup table.
///
/// Input: `rule`, u8, range [0, 255].
/// Output: `[u8; 8]` where index = 3-bit neighborhood, value = 0 or 1.
/// Index encoding: `(left << 2) | (center << 1) | right`.
/// Each bit `i` of the rule number determines `table[i]`.
///
/// Example: rule 30 (binary 00011110) produces [0, 1, 1, 1, 1, 0, 0, 0].
///
/// Does not validate the rule number — all 256 values of u8 are valid Wolfram rules.
fn build_rule_table(rule: u8) -> [u8; 8] {
    let mut table = [0u8; 8];
    for i in 0..8u8 {
        table[i as usize] = (rule >> i) & 1;
    }
    table
}

/// Sets a single bit in a bitpacked row buffer.
///
/// Input: `row` is a mutable slice of u64 words, MSB-first packing.
/// Input: `pos` is the cell index (0-based from left).
/// Mapping: word index = pos / 64, bit index = 63 - (pos % 64).
///
/// Does not bounds-check `pos` against logical width — caller must ensure pos < width.
#[inline]
fn set_bit(row: &mut [u64], pos: usize) {
    let word = pos / 64;
    let bit = 63 - (pos % 64);
    row[word] |= 1u64 << bit;
}

/// Computes one generation of the cellular automaton using bitpacked rows.
///
/// Input: `current` — the current generation as bitpacked u64 words, MSB-first.
/// Output: `next` — the next generation, written into this pre-zeroed buffer.
/// Input: `rule_table` — 8-entry lookup from `build_rule_table`.
/// Input: `width` — logical number of cells. May be less than current.len() * 64.
///
/// Boundary condition: cells at positions < 0 or >= width are treated as 0 (dead).
///
/// Algorithm per word:
/// 1. Extract left-neighbor bits: shift current word right by 1, splice in bit 0
///    of the previous word at bit 63.
/// 2. Extract right-neighbor bits: shift current word left by 1, splice in bit 63
///    of the next word at bit 0.
/// 3. For each bit position in the word (up to `width`), assemble the 3-bit
///    neighborhood (left, center, right) and look up the new state in rule_table.
///
/// Known limitation: the per-bit loop (step 3) could be replaced with fully bitwise
/// boolean logic for higher throughput. Current approach is correct but not maximally fast.
fn compute_next_generation(
    current: &[u64],
    next: &mut [u64],
    rule_table: &[u8; 8],
    width: usize,
) {
    let num_words = current.len();

    for wi in 0..num_words {
        let cell_start = wi * 64;
        if cell_start >= width {
            break;
        }

        let cur = current[wi];
        // prev_word: the word to the left. 0 if wi == 0 (left boundary is dead).
        let prev_word = if wi > 0 { current[wi - 1] } else { 0 };
        // next_word: the word to the right. 0 if last word (right boundary is dead).
        let next_word = if wi + 1 < num_words { current[wi + 1] } else { 0 };

        // Bit b in left_neighbors = left neighbor of the cell at bit b in cur.
        // Left neighbor of bit b is at bit b+1 in cur, except bit 63 comes from prev_word's bit 0.
        let left_neighbors = (cur >> 1) | ((prev_word & 1) << 63);
        // Bit b in right_neighbors = right neighbor of the cell at bit b in cur.
        // Right neighbor of bit b is at bit b-1 in cur, except bit 0 comes from next_word's bit 63.
        let right_neighbors = (cur << 1) | (next_word >> 63);

        let mut result: u64 = 0;

        // Process only valid cells (up to `width`, not full 64 bits in the last word).
        let cell_end = std::cmp::min(cell_start + 64, width);
        for pos in cell_start..cell_end {
            let b = 63 - (pos % 64);

            let left = (left_neighbors >> b) & 1;
            let center = (cur >> b) & 1;
            let right = (right_neighbors >> b) & 1;

            let neighborhood = ((left << 2) | (center << 1) | right) as usize;
            let new_state = rule_table[neighborhood] as u64;
            result |= new_state << b;
        }

        next[wi] = result;
    }
}

/// Decodes a bitpacked row and writes live-cell pixels into the flat image buffer.
///
/// Input: `row` — bitpacked u64 words, MSB-first.
/// Input: `img_buf` — flat pixel buffer, row-major, 1 byte per pixel. Must be pre-filled
///        with 0xFF (white/dead): only live cells are written, dead cells are skipped.
/// Input: `row_idx` — which row in the image (0-based). Pixels start at img_buf[row_idx * width].
/// Input: `width` — logical number of cells/pixels per row.
///
/// Pixel mapping: bit value 1 (live) → 0x00 (black). Bit value 0 (dead) → untouched
/// (stays 0xFF from the caller's pre-fill).
///
/// Does not bounds-check row_idx — caller must ensure row_idx * width + width <= img_buf.len().
fn decode_row_to_image(
    row: &[u64],
    img_buf: &mut [u8],
    row_idx: usize,
    width: usize,
) {
    let row_offset = row_idx * width;
    for (wi, &word) in row.iter().enumerate() {
        let cell_start = wi * 64;
        if cell_start >= width {
            break;
        }
        // All-dead words need no writes at all — rows are sparse for most rules.
        if word == 0 {
            continue;
        }
        let cell_end = std::cmp::min(cell_start + 64, width);
        for pos in cell_start..cell_end {
            let b = 63 - (pos % 64);
            if (word >> b) & 1 == 1 {
                img_buf[row_offset + pos] = 0x00;
            }
        }
    }
}

// ==========================================================================
// Tests: naive reference implementation + correctness verification
// ==========================================================================

/// Naive (non-bitpacked) implementation used only in tests as a correctness reference.
/// This module is compiled only under `#[cfg(test)]` — it is not included in the release binary.
#[cfg(test)]
mod naive {
    use super::build_rule_table;

    /// Simulates a 1D elementary cellular automaton without bitpacking.
    ///
    /// Input: `rule`, u8 — Wolfram rule number.
    /// Input: `width`, usize — number of cells per row.
    /// Input: `steps`, usize — number of generations to compute.
    /// Output: Vec<Vec<u8>> — rows[0] is the initial state, rows[steps] is the final generation.
    ///         Each inner Vec has length `width`. Values are 0 (dead) or 1 (alive).
    ///
    /// Initial state: all cells 0 except center cell (index width / 2) set to 1.
    /// Boundary condition: cells at index < 0 or >= width are 0 (dead).
    pub fn simulate_naive(rule: u8, width: usize, steps: usize) -> Vec<Vec<u8>> {
        let rule_table = build_rule_table(rule);
        let mut rows = Vec::with_capacity(steps + 1);

        let mut current = vec![0u8; width];
        current[width / 2] = 1;
        rows.push(current.clone());

        for _ in 0..steps {
            let mut next = vec![0u8; width];
            for i in 0..width {
                let left = if i == 0 { 0u8 } else { current[i - 1] };
                let center = current[i];
                let right = if i + 1 >= width { 0u8 } else { current[i + 1] };
                let neighborhood =
                    ((left as usize) << 2) | ((center as usize) << 1) | (right as usize);
                next[i] = rule_table[neighborhood];
            }
            rows.push(next.clone());
            current = next;
        }

        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::naive::simulate_naive;

    /// Converts a bitpacked row (Vec<u64>, MSB-first) into a Vec<u8> of 0s and 1s.
    /// Used to compare bitpacked output against the naive implementation's output format.
    fn bitpacked_to_vec(row: &[u64], width: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(width);
        for pos in 0..width {
            let word = pos / 64;
            let bit = 63 - (pos % 64);
            result.push(((row[word] >> bit) & 1) as u8);
        }
        result
    }

    /// Runs the bitpacked simulation and returns all rows as Vec<Vec<u8>> (same format as naive).
    /// This duplicates main()'s simulation logic so tests can compare row-by-row without PNG I/O.
    fn simulate_bitpacked(rule: u8, width: usize, steps: usize) -> Vec<Vec<u8>> {
        let rule_table = build_rule_table(rule);
        let num_words = width.div_ceil(64);
        let mut current = vec![0u64; num_words];
        let mut next = vec![0u64; num_words];

        set_bit(&mut current, width / 2);

        let mut rows = Vec::with_capacity(steps + 1);
        rows.push(bitpacked_to_vec(&current, width));

        for _ in 0..steps {
            compute_next_generation(&current, &mut next, &rule_table, width);
            rows.push(bitpacked_to_vec(&next, width));
            std::mem::swap(&mut current, &mut next);
            next.fill(0);
        }

        rows
    }

    /// Compares naive and bitpacked implementations row-by-row for a given rule, width, and steps.
    /// Panics with a descriptive message on first mismatch, including rule, width, and generation.
    fn compare_implementations(rule: u8, width: usize, steps: usize) {
        let naive_rows = simulate_naive(rule, width, steps);
        let bitpacked_rows = simulate_bitpacked(rule, width, steps);

        assert_eq!(
            naive_rows.len(),
            bitpacked_rows.len(),
            "Rule {}, width {}: row count mismatch",
            rule,
            width
        );

        for (row_idx, (naive_row, bp_row)) in
            naive_rows.iter().zip(bitpacked_rows.iter()).enumerate()
        {
            assert_eq!(
                naive_row, bp_row,
                "Rule {}, width {}, generation {}: mismatch",
                rule, width, row_idx
            );
        }
    }

    // --- Correctness tests: compare naive vs bitpacked for 200 generations ---
    // Width = 2 * 200 + 1 = 401 (default triangle width).

    #[test]
    fn test_rule30_200_steps() {
        compare_implementations(30, 2 * 200 + 1, 200);
    }

    #[test]
    fn test_rule90_200_steps() {
        compare_implementations(90, 2 * 200 + 1, 200);
    }

    #[test]
    fn test_rule110_200_steps() {
        compare_implementations(110, 2 * 200 + 1, 200);
    }

    // --- Word-boundary tests ---
    // These widths are chosen to stress u64 word boundaries:
    // 64 = exactly 1 word (no adjacent word on either side), 65 = 1 word + 1 bit,
    // 127 = 2 words - 1 bit, 128 = exactly 2 words, 129 = 2 words + 1 bit.
    // If cross-word bit splicing is wrong, these tests will catch it.

    #[test]
    fn test_rule30_width_64() {
        compare_implementations(30, 64, 200);
    }

    #[test]
    fn test_rule30_width_65() {
        compare_implementations(30, 65, 200);
    }

    #[test]
    fn test_rule30_width_127() {
        compare_implementations(30, 127, 200);
    }

    #[test]
    fn test_rule30_width_128() {
        compare_implementations(30, 128, 200);
    }

    #[test]
    fn test_rule30_width_129() {
        compare_implementations(30, 129, 200);
    }

    #[test]
    fn test_rule90_width_65() {
        compare_implementations(90, 65, 200);
    }

    #[test]
    fn test_rule110_width_129() {
        compare_implementations(110, 129, 200);
    }

    // --- Edge case tests ---

    /// Rule 0: every neighborhood maps to 0. After generation 0, all cells must be dead.
    #[test]
    fn test_rule0_all_dead() {
        let width = 65;
        let naive = simulate_naive(0, width, 10);
        let bp = simulate_bitpacked(0, width, 10);
        for i in 1..=10 {
            assert!(naive[i].iter().all(|&c| c == 0));
            assert!(bp[i].iter().all(|&c| c == 0));
        }
    }

    /// Rule 255: every neighborhood maps to 1. Live cells expand by 1 on each side per generation.
    #[test]
    fn test_rule255_all_alive() {
        compare_implementations(255, 21, 5);
    }

    /// Width 1: single cell, no neighbors. Tests minimal allocation and boundary-only behavior.
    #[test]
    fn test_small_width_1() {
        compare_implementations(30, 1, 10);
    }

    /// Width 3: center cell has exactly one neighbor on each side. Minimal non-trivial case.
    #[test]
    fn test_small_width_3() {
        compare_implementations(30, 3, 10);
    }

    /// Verifies the rule table bit extraction for known rules.
    /// Rule 30 = 0b00011110 → [0, 1, 1, 1, 1, 0, 0, 0].
    /// Rule 90 = 0b01011010 → [0, 1, 0, 1, 1, 0, 1, 0].
    /// Rule 110 = 0b01101110 → [0, 1, 1, 1, 0, 1, 1, 0].
    #[test]
    fn test_rule_table_correctness() {
        let table = build_rule_table(30);
        assert_eq!(table, [0, 1, 1, 1, 1, 0, 0, 0]);

        let table = build_rule_table(90);
        assert_eq!(table, [0, 1, 0, 1, 1, 0, 1, 0]);

        let table = build_rule_table(110);
        assert_eq!(table, [0, 1, 1, 1, 0, 1, 1, 0]);
    }

    /// Exhaustive sweep: every one of the 256 Wolfram rules, naive vs bitpacked.
    /// Catches rule-table or neighborhood-encoding errors in rules the sampled tests skip.
    /// Width 65 spans two words so cross-word splicing runs for every rule; 32 steps lets
    /// the pattern reach both edges from the center (cell 32, growth 1 cell per side per
    /// generation), exercising both boundary conditions.
    #[test]
    fn test_all_256_rules() {
        for rule in 0..=255u8 {
            compare_implementations(rule, 65, 32);
        }
    }

    /// Verifies decode_row_to_image pixel values, row offset, and word-boundary mapping.
    /// Failure modes covered: inverted color mapping, wrong row offset (writes landing in
    /// adjacent image rows), and bit-position errors at the word 0 / word 1 boundary.
    /// Live cells 0, 63, 64, 69: first and last bit of word 0, first bit of word 1, and
    /// the last valid cell of a partial word (width 70).
    #[test]
    fn test_decode_row_to_image() {
        let width = 70;
        let live = [0usize, 63, 64, 69];
        let mut row = vec![0u64; 2];
        for &pos in &live {
            set_bit(&mut row, pos);
        }

        // 3-row buffer, decoding into the middle row, so an offset error in either
        // direction lands in a row asserted untouched below.
        let mut img_buf = vec![0xFFu8; width * 3];
        decode_row_to_image(&row, &mut img_buf, 1, width);

        for pos in 0..width {
            let expected = if live.contains(&pos) { 0x00 } else { 0xFF };
            assert_eq!(img_buf[width + pos], expected, "pixel {} in decoded row", pos);
        }
        assert!(img_buf[..width].iter().all(|&p| p == 0xFF), "row 0 must be untouched");
        assert!(img_buf[width * 2..].iter().all(|&p| p == 0xFF), "row 2 must be untouched");
    }
}
