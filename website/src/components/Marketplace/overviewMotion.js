// Scroll-driven placement in reserved image lanes; the logo artwork stays intact.
const clamp = (value) => Math.max(0, Math.min(1, value));
const ease = (value) => {
  const t = clamp(value);
  return t * t * (3 - 2 * t);
};
const mix = (a, b, t) => a + (b - a) * t;

export function logoPosition(
  {hero, chapters, height, width, header},
  scrollY,
  wide,
) {
  if (!chapters.length) return null;
  if (!wide) {
    const slots = [hero, ...chapters.map((chapter) => chapter.slot)];
    const candidates = slots
      .map((slot, index) => ({
        slot,
        index,
        visible: Math.max(
          0,
          Math.min(slot.bottom - scrollY, height) -
            Math.max(slot.top - scrollY, header),
        ),
      }))
      .filter(({slot, visible}) => visible >= Math.min(slot.height * 0.35, 64));
    candidates.sort(
      (a, b) =>
        Math.abs(a.slot.y - scrollY - height * 0.45) -
        Math.abs(b.slot.y - scrollY - height * 0.45),
    );
    if (!candidates.length) return null;
    const {slot, index, visible} = candidates[0];
    const size = Math.min(240, width * 0.58, slot.width, slot.height);
    let progress = ease(
      ((scrollY + height - slot.top) / (height + slot.height) - 0.2) / 0.6,
    );
    if (index % 2) progress = 1 - progress;
    return {
      x: slot.left + size / 2 + (slot.width - size) * progress,
      y: slot.y - scrollY,
      size,
      opacity: ease((visible / slot.height - 0.35) / 0.4),
    };
  }
  const focus = Math.max(scrollY + height * 0.45, hero.y);
  if (focus >= chapters.at(-1).bottom) return null;
  const first = chapters[0];
  const introduction = ease(
    (focus - hero.y) / Math.max(1, first.top + first.padding - hero.y),
  );
  const firstSize = Math.min(300, first.slot.width);
  if (introduction < 1) {
    return {
      x: mix(hero.x, first.slot.x, introduction),
      y: focus - scrollY,
      size: mix(hero.width, firstSize, introduction),
      opacity: 1,
    };
  }
  const chapter = chapters.findLast((item) => item.top <= focus) || first;
  let x = chapter.slot.x;
  let size = Math.min(300, chapter.slot.width);
  const boundary = chapters
    .slice(1)
    .find((item) => Math.abs(focus - item.top) < item.padding * 1.4);
  if (boundary) {
    const previous = chapters[chapters.indexOf(boundary) - 1];
    const offset = focus - boundary.top;
    const bridge = Math.max(1, boundary.padding * 0.4);
    x = mix(
      previous.slot.x,
      boundary.slot.x,
      ease((offset + bridge) / (bridge * 2)),
    );
    size = mix(
      Math.min(size, boundary.padding),
      size,
      ease(Math.abs(offset) / (boundary.padding * 1.4)),
    );
  }
  const fade = clamp((chapters.at(-1).bottom - focus) / Math.max(1, size));
  return {x, y: focus - scrollY, size, opacity: fade};
}

export function attachLogoJourney(root, image) {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const wide = window.matchMedia('(min-width: 1001px) and (min-height: 651px)');
  let geometry;
  let frame = 0;
  let dirty = true;
  const position = (element) => {
    const r = element.getBoundingClientRect();
    return {
      left: r.left,
      top: r.top + window.scrollY,
      bottom: r.bottom + window.scrollY,
      width: r.width,
      height: r.height,
      x: r.left + r.width / 2,
      y: r.top + window.scrollY + r.height / 2,
    };
  };
  function render() {
    frame = 0;
    if (reduced.matches) {
      delete root.dataset.logoJourney;
      image.hidden = true;
      return;
    }
    if (dirty) {
      geometry = {
        hero: position(root.querySelector('[data-hero-logo] [data-logo-slot]')),
        chapters: [...root.querySelectorAll('[data-logo-chapter]')].map(
          (element) => ({
            ...position(element),
            padding: parseFloat(getComputedStyle(element).paddingTop),
            slot: position(element.querySelector('[data-logo-slot]')),
          }),
        ),
        height: window.innerHeight,
        width: window.innerWidth,
        header:
          document.querySelector('.navbar')?.getBoundingClientRect().height ||
          0,
      };
      dirty = false;
    }
    const point = logoPosition(geometry, window.scrollY, wide.matches);
    root.dataset.logoJourney = 'active';
    image.hidden = !point;
    if (point) {
      image.style.transform = `translate3d(${point.x - point.size / 2}px, ${point.y - point.size / 2}px, 0) scale(${point.size / 440})`;
      image.style.opacity = point.opacity;
    }
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(render);
  };
  const measure = () => {
    dirty = true;
    schedule();
  };
  window.addEventListener('scroll', schedule, {passive: true});
  window.addEventListener('resize', measure);
  reduced.addEventListener('change', measure);
  wide.addEventListener('change', measure);
  const observer = new ResizeObserver(measure);
  observer.observe(root);
  measure();
  return () => {
    cancelAnimationFrame(frame);
    observer.disconnect();
    window.removeEventListener('scroll', schedule);
    window.removeEventListener('resize', measure);
    reduced.removeEventListener('change', measure);
    wide.removeEventListener('change', measure);
    delete root.dataset.logoJourney;
  };
}
