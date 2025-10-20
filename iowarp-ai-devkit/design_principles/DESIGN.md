# IoWarp Design System - Neobrutalist Principles

This document captures the official design language for IoWarp projects, particularly the MCP website. These principles guide all visual design decisions and ensure consistency across future development.

**Last Updated:** October 19, 2025
**Status:** Finalized and Locked

---

## Design Philosophy

IoWarp's visual identity is built on **neobrutalism** - a design movement emphasizing bold structure, raw materials, and functional honesty. Our interpretation focuses on:

1. **Structure over decoration** - Design emerges from layout and typography, not superficial styling
2. **Dark-first aesthetic** - Embracing deep, technical darkness with careful accent usage
3. **Functional brutality** - Every element serves a purpose; no decoration for decoration's sake
4. **Science meets craft** - Technical precision balanced with human warmth

> "Neobrutalism from shapes and shadows, not color contrasts"

---

## Color Science

### Locked Brand Palette

These four colors are **sacred** and derived directly from the IoWarp logo. Never modify these values:

```css
--brand-teal: #217CA3;          /* Primary brand color */
--brand-teal-light: #6BC2E4;    /* Light accent variant */
--brand-orange: #EC7C26;        /* Energy and warmth */
--brand-cream: #FAF8F5;         /* Warmth in light mode */
```

**Color Usage Rules:**
- **Teal family**: Primary brand identity, links, highlights
- **Orange**: Energy, call-to-action, emphasis
- **Cream**: Light mode warmth (never in dark mode)
- **NEVER** use these for large background areas in dark mode

### Dark Mode Color System (Default)

Dark mode is the **primary** experience. All design starts here.

```css
--color-bg: #000000;              /* Pure black - no compromise */
--color-surface: #0F1F35;         /* Dark blue for cards/elements */
--color-text: #E5E7EB;            /* Light gray for readability */
--color-text-muted: #9CA3AF;      /* Secondary text */
--color-border: rgba(107, 194, 228, 0.25);  /* Subtle teal borders */
--color-shadow: rgba(107, 194, 228, 0.4);   /* CRITICAL: Light teal shadows! */
```

**Key Principle:** Shadows must be **light teal** in dark mode for visibility. Black shadows disappear against black backgrounds.

**Shadow Tokens:**
```css
--shadow-sm: 4px 4px 0px 0px rgba(107, 194, 228, 0.3);
--shadow-md: 6px 6px 0px 0px rgba(107, 194, 228, 0.4);
--shadow-lg: 8px 8px 0px 0px rgba(107, 194, 228, 0.5);
--shadow-xl: 10px 10px 0px 0px rgba(107, 194, 228, 0.6);
```

### Light Mode Color System

Light mode offers a **warm alternative** to dark mode's intensity.

```css
--color-bg: #FAF8F5;              /* Warm cream, not cold white */
--color-surface: #FFFFFF;         /* Clean white for cards */
--color-text: #111827;            /* Nearly black for contrast */
--color-text-muted: #6B7280;      /* Warm gray for secondary */
--color-border: #111827;          /* Dark borders */
--color-shadow: #111827;          /* Dark shadows for visibility */
```

**Key Difference:** Light mode uses **black shadows** for visibility. Opposite of dark mode.

### Semantic Category Colors

These colors carry **meaning** and remain consistent across themes:

```javascript
categories = {
  "Formats": "#217CA3",     // Teal - Data structures
  "Analytics": "#6BC2E4",   // Light teal - Processing
  "HPC": "#EC7C26",         // Orange - High performance
  "Performance": "#10b981", // Green - Speed/efficiency
  "Research": "#8B5CF6",    // Purple - Academic/scholarly
  "Utilities": "#6b7280"    // Gray - Tools/helpers
}
```

These colors are **independent of theme** - they communicate meaning, not aesthetic preference.

---

## Neobrutalist Structure

### The Golden Rules

1. **Single border width**: 3px everywhere (`--border-width: 3px`)
2. **Single border radius**: 8px everywhere (`--border-radius: 8px`)
3. **Hard shadows only**: No blur, no gradients
4. **Bold typography**: Minimum 700 weight for headings, 900 for hero
5. **Functional transforms**: Rotation/skew only when it enhances interaction

### Border & Shadow Pattern

```css
/* Correct neobrutalist shadow */
box-shadow: 6px 6px 0px 0px var(--color-shadow);

/* NEVER do this */
box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.3);  /* ❌ Blur = not brutal */
```

**Shadow Scale:**
- **4px**: Small elements (chips, badges)
- **6px**: Medium elements (cards, buttons)
- **8px**: Large elements (search bar, modals)
- **10px**: Extra large (hero elements, emphasis)

### Typography System

```css
--font-display: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
--font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
```

**Hierarchy:**
- **Hero titles**: 900 weight, tight letter-spacing (-0.02em)
- **Section headers**: 700 weight, uppercase with wide spacing (0.08-0.12em)
- **Body text**: 400-500 weight, comfortable line-height (1.6-1.8)
- **UI labels**: 700 weight, uppercase, very wide spacing (0.12em+)

### Animation Philosophy

Animations should be **purposeful and subtle**:

```css
/* Hover transform - adds energy */
transform: translate(-4px, -4px) rotate(-1deg);

/* NEVER overdo it */
transform: translate(-20px, -20px) rotate(-15deg);  /* ❌ Too dramatic */
```

**Approved Transforms:**
- Rotation: -2deg to 2deg
- Skew: -3deg to 3deg
- Translate: 2-6px on hover (matching shadow growth)
- Transition: 150ms ease (never longer than 200ms)

---

## Component Design Patterns

### Hero Section

The hero is the **most important** design element - it sets the tone.

**Structure:**
```
Hero (Teal background with millimeter grid)
├── Gnosis Badge (floating, rotated -1.5deg)
├── Logo + Brand Text (horizontal, centered)
├── Title (bold, white, dramatic)
├── CTA Buttons (3 types: primary/outline/ghost)
├── Subtitle (with inline SEO links)
├── Value Cards (3 cards, inside hero)
└── Footer Attribution (NSF credit)
```

**Millimeter Grid Pattern:**
```css
background-image:
  linear-gradient(rgba(248, 176, 119, 0.35) 1px, transparent 1px),
  linear-gradient(90deg, rgba(248, 176, 119, 0.35) 1px, transparent 1px);
background-size: 20px 20px;
```

- Light mode: Orange grid at 35% opacity
- Dark mode: Light teal grid at 35% opacity
- Grid size: Always 20px × 20px

**Gnosis Badge Rules:**
```css
/* Light mode */
background: #FFFFFF;
border: 4px solid #0A1520;
box-shadow: 6px 6px 0px 0px #0A1520;

/* Dark mode */
background: #0F1F35;
border: 4px solid var(--brand-orange);
box-shadow: 6px 6px 0px 0px var(--brand-orange);
```

**Critical:** Background is always light/dark, outline is always the opposite theme color.

### MCP Cards

Two card types serve different purposes:

**Featured Cards (3-4 per page):**
- Large format with detailed information
- Logo + name + version + GitHub link
- 4 tags: Endorsement, Type, Tool Count, Special
- Full description
- Platform integration buttons (Claude, Cursor, VSCode, Gemini)
- Hover: Subtle lift with shadow growth

**Regular Cards (catalog view):**
- Compact, scannable
- Icon + category tag in header
- Name + short description
- Single "View Details" button
- Grid layout: 3 columns on desktop, 1 on mobile

**Card Anatomy:**
```css
.card {
  border: 3px solid var(--color-border);
  border-radius: 8px;
  box-shadow: var(--shadow-md);
  background: var(--color-surface);  /* #0F1F35 in dark, #FFFFFF in light */
}
```

### Search & Filter Interface

**Search Bar:**
```css
.searchBox {
  transform: rotate(-0.5deg);  /* Subtle warp effect */
  box-shadow: 8px 8px 0px 0px var(--color-black);
}

[data-theme='dark'] .searchBox {
  box-shadow: 8px 8px 0px 0px rgba(236, 124, 38, 0.4);  /* Orange glow! */
}
```

**Category Chips:**
- Filled backgrounds using semantic category colors
- Active state: 3px border + slight scale
- Uppercase text, 700 weight
- Count badge on right side

---

## Critical Anti-Patterns

### ❌ Things to NEVER Do

1. **White boxes on dark backgrounds**
   ```css
   /* WRONG */
   [data-theme='dark'] .card {
     background: #FFFFFF;  /* ❌ Breaks cohesion */
   }

   /* CORRECT */
   [data-theme='dark'] .card {
     background: #0F1F35;  /* ✓ Maintains dark aesthetic */
   }
   ```

2. **Black shadows in dark mode**
   ```css
   /* WRONG */
   [data-theme='dark'] {
     --shadow: 6px 6px 0px 0px #000000;  /* ❌ Invisible! */
   }

   /* CORRECT */
   [data-theme='dark'] {
     --shadow: 6px 6px 0px 0px rgba(107, 194, 228, 0.4);  /* ✓ Visible! */
   }
   ```

3. **Multiple border radii**
   ```css
   /* WRONG */
   --border-radius-sm: 4px;
   --border-radius-md: 8px;
   --border-radius-lg: 12px;

   /* CORRECT */
   --border-radius: 8px;  /* ✓ Single value everywhere */
   ```

4. **Gradient backgrounds**
   ```css
   /* WRONG */
   background: linear-gradient(135deg, #217CA3, #EC7C26);  /* ❌ Too soft */

   /* CORRECT */
   background: var(--brand-teal);  /* ✓ Flat, honest */
   ```

5. **Soft shadows**
   ```css
   /* WRONG */
   box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);  /* ❌ Not brutal */

   /* CORRECT */
   box-shadow: 8px 8px 0px 0px var(--color-shadow);  /* ✓ Hard edge */
   ```

---

## Responsive Design

### Breakpoints

```css
--mobile: 768px;
--tablet: 996px;
--desktop: 1200px;
```

**Mobile-First Approach:**
```css
/* Mobile default */
.grid {
  grid-template-columns: 1fr;
}

/* Tablet and up */
@media (min-width: 768px) {
  .grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Desktop */
@media (min-width: 996px) {
  .grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
```

### Mobile Adaptations

- **Hero logo**: 280px on mobile (from 360px)
- **Hero title**: 2rem on mobile (from 2.5rem)
- **Card grid**: 1 column (from 3)
- **Search bar**: Full width, no rotation
- **Category chips**: Scroll horizontally

---

## Accessibility

Despite the brutal aesthetic, accessibility is **non-negotiable**:

### Contrast Requirements

- **Normal text**: Minimum 4.5:1 contrast ratio
- **Large text**: Minimum 3:1 contrast ratio
- **UI elements**: Minimum 3:1 against background

Our dark mode palette:
- `#E5E7EB` on `#000000` = 15.5:1 ✓
- `#E5E7EB` on `#0F1F35` = 10.8:1 ✓
- `#6BC2E4` on `#0F1F35` = 6.2:1 ✓

Our light mode palette:
- `#111827` on `#FAF8F5` = 14.2:1 ✓
- `#217CA3` on `#FFFFFF` = 4.9:1 ✓

### Interaction States

All interactive elements must have:
- **Hover**: Visual feedback (shadow growth, transform)
- **Focus**: Clear outline (2px solid, offset 2px)
- **Active**: Pressed state (shadow reduction)
- **Disabled**: 50% opacity + cursor: not-allowed

```css
button:focus-visible {
  outline: 2px solid var(--brand-orange);
  outline-offset: 2px;
}
```

---

## Logo & Asset Guidelines

### Main Logo

- **File**: `iowarp_logo.png`
- **Size**: 360px × auto (desktop), 280px × auto (mobile)
- **Format**: PNG with transparency
- **Usage**: Hero section only

### Platform Logos

Located in `static/img/logos/`:
- `claude-logo.png` - Claude AI
- `cursor-logo.png` - Cursor IDE
- `vscode-logo.png` - VS Code
- `gemini-logo.png` - Google Gemini

**Size**: 24px × 24px
**Format**: PNG
**Usage**: Platform integration buttons

### Institution Logos

- `grc-logo.png` - Gnosis Research Center (24px)
- `iit-logo.png` - Illinois Tech (56px)
- `nsf-logo.png` - National Science Foundation (24px)

**Treatment**: Display in original colors (no filters in dark mode)

### MCP Server Logos

Each MCP gets a logo in both SVG and PNG:
- Primary: SVG for scalability
- Fallback: PNG if SVG fails
- Size: 48px × 48px in featured cards
- Size: 32px × 32px in regular cards

**Naming**: `{mcp-id}-logo.svg` or `{mcp-id}-logo.png`

---

## Theme Toggle Behavior

Users can switch between dark and light modes. The toggle:

- **Default**: Dark mode
- **Persistence**: LocalStorage saves preference
- **Respect System**: `respectPrefersColorScheme: false` (we default to dark)
- **Animation**: Smooth 300ms transition on theme change

```css
* {
  transition: background-color 300ms ease,
              color 300ms ease,
              border-color 300ms ease;
}
```

---

## SEO & Metadata Strategy

### Comprehensive Coverage

Every page includes:
- **Open Graph**: Social sharing optimization
- **Twitter Cards**: Enhanced link previews
- **Structured Data**: JSON-LD for search engines
- **Keywords**: Targeted scientific computing terms
- **Descriptions**: Unique, keyword-rich summaries

### Official Technology Links

Inline links to official documentation:
- HDF5 → https://www.hdfgroup.org/solutions/hdf5/
- ADIOS → https://adios2.readthedocs.io/
- Slurm → https://slurm.schedmd.com/
- Pandas → https://pandas.pydata.org/
- And 10+ more scientific tools

**Purpose:** SEO authority + user education

---

## File Organization

```
iowarp_mcp_webpage/
├── src/
│   ├── css/
│   │   └── custom.css              # Global tokens, hero, theme system
│   ├── pages/
│   │   └── index.js                # Homepage hero and layout
│   ├── components/
│   │   ├── MCPShowcase/
│   │   │   ├── index.js            # Cards, search, filters
│   │   │   └── styles.module.css   # Component-specific styles
│   │   └── MCPDetail/
│   │       ├── index.js            # Individual MCP pages
│   │       └── styles.module.css
│   └── data/
│       └── mcpData.js              # MCP metadata, categories
├── static/
│   └── img/
│       ├── logos/                  # All logo assets
│       └── iowarp_logo.png         # Main brand asset
├── docusaurus.config.js            # Site configuration
├── package.json                    # Dependencies
└── .gitignore                      # Excludes build/, node_modules/
```

**Key Files for Design:**
- `src/css/custom.css` - All design tokens and hero styling
- `src/components/MCPShowcase/styles.module.css` - Component styling
- `docusaurus.config.js` - Theme configuration

---

## Development Workflow

### Local Development

```bash
cd iowarp_mcp_webpage
npm start              # Dev server at localhost:3000
```

### Production Build

```bash
npm run build         # Creates optimized build/
npm run serve         # Preview build locally
```

### Testing Checklist

Before committing design changes:

- [ ] Test in both dark and light modes
- [ ] Verify shadows are visible in both themes
- [ ] Check mobile responsiveness (DevTools)
- [ ] Validate all interactive states (hover, focus, active)
- [ ] Run accessibility audit (Lighthouse)
- [ ] Verify logo loading (SVG with PNG fallback)
- [ ] Test with slow connection (throttling)

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| Oct 19, 2025 | 1.0 | Initial design system finalized |

---

## Future Considerations

As the project evolves, consider:

1. **Animation library**: Framer Motion for advanced interactions
2. **Component library**: Extract to shared package if multiple sites
3. **Dark mode variants**: Add high-contrast mode for accessibility
4. **Print styles**: For documentation pages
5. **Theme customization**: User-selectable accent colors

But always maintain the core neobrutalist principles:
- Structure over decoration
- Honest materials
- Functional beauty
- Dark-first aesthetic

---

**Remember:** This design system is not about following trends. It's about creating a visual language that reflects IoWarp's mission: powerful, technical, honest, and a bit rebellious.

> "Neobrutalism from shapes and shadows, not color contrasts"
