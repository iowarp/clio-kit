const fs = require('node:fs/promises');
const path = require('node:path');

const entities = {amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' '};
const decode = (text) =>
  text.replace(/&(#x[\da-f]+|#\d+|\w+);/gi, (match, code) =>
    code[0] === '#'
      ? String.fromCodePoint(
          code[1].toLowerCase() === 'x'
            ? parseInt(code.slice(2), 16)
            : Number(code.slice(1)),
        )
      : (entities[code.toLowerCase()] ?? match),
  );

async function readPage(outDir, pathname) {
  const base = path.join(outDir, pathname);
  for (const file of [`${base}.html`, path.join(base, 'index.html')]) {
    const html = await fs.readFile(file, 'utf8').catch(() => null);
    if (html !== null) return html;
  }
  return '';
}

// Listed in output order; the first matching section wins.
const isMcp = (p) => p.startsWith('/docs/mcps/');

const sections = [
  ['Docs', (p) => (p === '/docs' || p.startsWith('/docs/')) && !isMcp(p)],
  ['MCP servers', (p) => isMcp(p)],
  ['Catalogue', (p) => p === '/catalogue' || p.startsWith('/catalogue/')],
  ['Optional', () => true],
];

/**
 * Writes /llms.txt from the exact URL list the sitemap plugin emits. Pass
 * `createSitemapItems` to the sitemap options and `plugin` to `plugins`.
 */
module.exports = function createLlmsTxt() {
  let resolveItems;
  let rejectItems;
  const sitemapItems = new Promise((resolve, reject) => {
    resolveItems = resolve;
    rejectItems = reject;
  });
  async function createSitemapItems({defaultCreateSitemapItems, ...params}) {
    try {
      const items = await defaultCreateSitemapItems(params);
      resolveItems(items);
      return items;
    } catch (err) {
      rejectItems(err);
      throw err;
    }
  }
  function plugin() {
    return {
      name: 'clio-llms-txt',
      async postBuild({siteConfig, outDir}) {
        if (siteConfig.noIndex) return;
        const delimiter = ` ${siteConfig.titleDelimiter} `;
        const groups = new Map(sections.map(([name]) => [name, []]));
        for (const {url} of await sitemapItems) {
          const pathname = new URL(url).pathname;
          const html = await readPage(outDir, pathname);
          let title = decode(html.match(/<title[^>]*>([^<]*)<\/title>/)?.[1] ?? '');
          if (title.endsWith(delimiter + siteConfig.title)) {
            title = title.slice(0, -(delimiter.length + siteConfig.title.length));
          }
          const description = decode(
            html.match(/<meta[^>]*name="description"[^>]*content="([^"]*)"/)?.[1] ??
              '',
          );
          const [name] = sections.find(([, match]) => match(pathname));
          groups
            .get(name)
            .push(
              `- [${title || pathname}](${url})${description ? `: ${description}` : ''}`,
            );
        }
        const body = [
          `# ${siteConfig.title}`,
          '',
          `> ${siteConfig.tagline} CLIO Kit installs maintained scientific MCP servers through one launcher, ships portable skills, and bundles them into workflow plugins for Claude Code, Codex, Cursor, VS Code / Copilot and other agent clients. MCP server pages document tools and prerequisites; catalogue pages give per-client install steps for every listed entry.`,
          ...[...groups].flatMap(([name, lines]) =>
            lines.length ? ['', `## ${name}`, '', ...lines] : [],
          ),
          '',
        ].join('\n');
        await fs.writeFile(path.join(outDir, 'llms.txt'), body);
      },
    };
  }
  return {plugin, createSitemapItems};
};
