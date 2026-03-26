# Design System Document

## 1. Overview & Creative North Star
The Creative North Star for this system is **"The Blueprint Ethos."** 

This system moves beyond mere minimalism into the realm of technical editorial design. It treats the digital interface not as a collection of buttons, but as a living engineering diagram—precise, authoritative, and intentionally raw. By combining high-contrast geometric headers with a clinical, monospaced data layer, we evoke the feeling of a high-end research publication. 

The aesthetic rejects "web-standard" fluff. We use an asymmetric layout logic where white space is treated as a structural component, and "Electric Blue" acts as the sole conductor of user attention. The result is a signature experience that feels "built" rather than "templated."

## 2. Colors
Our palette is rooted in a clinical, high-key environment. The goal is to maximize the "glow" of the primary electric blue against a sophisticated hierarchy of near-white surfaces.

*   **Primary (`#0050cb`):** The "Electric Signal." Used exclusively for interactive elements, key data highlights, and architectural accents.
*   **Surface Hierarchy:**
    *   **Surface (`#f8f9fa`):** The base canvas.
    *   **Surface-Container-Low (`#f3f4f5`):** Used for subtle grouping.
    *   **Surface-Container-Highest (`#e1e3e4`):** Used for nested technical panels.
*   **The "No-Line" Rule:** 1px solid borders are strictly prohibited for sectioning. Boundaries must be defined through background color shifts. For instance, a sidebar should be differentiated from the main content solely by moving from `surface` to `surface-container-low`.
*   **The "Glass & Gradient" Rule:** To provide a premium "lens" effect, floating panels (like tooltips or overlays) should use `surface-container-lowest` with a 0.8 opacity and a 12px backdrop-blur. 
*   **Signature Textures:** Use a subtle linear gradient from `primary` to `primary-container` on high-level data visualizations or primary CTAs to add "spectral depth" that mimics a light-emitting diode.

## 3. Typography
The typographic system is a dialogue between the human (Sans-Serif) and the machine (Monospaced).

*   **Display & Headline (Space Grotesk):** These are your "Architectural Markers." Use high-scale weights for major headings to create a brutalist, editorial impact.
*   **Body (Inter):** Reserved for long-form reading and explanatory text. It provides the necessary legibility to balance the technical nature of the system.
*   **Title, Label, & Data (ui-monospace / Space Grotesk):** All technical data, metadata (e.g., timestamps, version numbers), and UI labels must use the monospaced scale. This reinforces the "Blueprint" aesthetic, making every piece of information feel like a calculated coordinate.

## 4. Elevation & Depth
In this system, depth is not "shadowed"; it is **layered.**

*   **The Layering Principle:** Avoid shadows to denote hierarchy. Instead, "stack" surface tiers. A `surface-container-lowest` card sitting on a `surface-container-low` section creates a clean, surgical lift.
*   **Ambient Shadows:** If a floating element (like a modal) requires a shadow, it must be nearly invisible. Use the `on-surface` color at 4% opacity with a 32px blur and 0px offset. It should feel like a soft atmospheric occlusion, not a drop shadow.
*   **The "Ghost Border" Fallback:** If a container requires a perimeter for legibility, use a "Ghost Border": `outline-variant` at 15% opacity. This maintains the "Blueprint" aesthetic without introducing heavy visual noise.
*   **Grid Integration:** Use subtle `px` lines or dotted `outline-variant` patterns in the background to mimic engineering grid paper, especially in hero sections or data-heavy views.

## 5. Components
Every component is an "Instrument."

*   **Buttons:** 
    *   **Primary:** Solid `primary` background, `on-primary` text. No rounded corners (`0px`).
    *   **Secondary:** Ghost-style. `outline-variant` (20% opacity) with `primary` text.
*   **Data Bars (Linear Progress):** As seen in the reference, these should be thick, pill-shaped markers within a `surface-container-high` track. Use `primary` for the active state to create a "liquid light" effect.
*   **Cards:** Forbid divider lines. Use vertical white space (`spacing-8` or `spacing-12`) or a background shift to `surface-container-lowest` to separate content blocks.
*   **Input Fields:** A simple `outline-variant` bottom-border (Ghost Border style). Labels should always be in `label-sm` (Monospaced) to feel like a field tag in a technical manual.
*   **Technical Chips:** Small, rectangular boxes with `surface-container-high` backgrounds and monospaced text. No rounding. Use them for tags like `V4.2.0-STABLE`.

## 6. Do's and Don'ts

### Do:
*   **Do** use intentional asymmetry. Align titles to the far left and data to the far right to create tension.
*   **Do** use the Spacing Scale rigorously. Large gaps of `spacing-24` between sections are necessary to let the "Blueprint" breathe.
*   **Do** treat data as the hero. Numbers should be larger and more prominent than the labels describing them.

### Don't:
*   **Don't** use border-radius. This system is defined by hard `0px` edges to maintain its technical integrity.
*   **Don't** use "Generic Grey" for text. Use `on-surface-variant` for a blue-tinted grey that feels integrated into the palette.
*   **Don't** use standard icons. Use thin-stroke (1px) geometric icons or monospaced characters (e.g., `[+]`, `->`, `::`) to maintain the "Making Software" aesthetic.
*   **Don't** use 100% opaque borders for containers. They break the fluid, layered feel of the "Blueprint."