const fs = require('node:fs/promises');
const path = require('node:path');

// Pinned community entries by upstream repository, so indexed items take their
// sitemap lastmod from the pin that determines their content.
async function communityEntries(repoRoot) {
  const dir = path.join(repoRoot, 'community/entries');
  const entries = new Map();
  for (const name of await fs.readdir(dir).catch(() => [])) {
    if (!name.endsWith('.toml')) continue;
    const file = path.join(dir, name);
    const repo = (await fs.readFile(file, 'utf8')).match(
      /^repo\s*=\s*"([^"]+)"/m,
    )?.[1];
    if (repo) entries.set(`https://github.com/${repo}`, file);
  }
  return entries;
}

async function exists(file) {
  return fs.access(file).then(
    () => true,
    () => false,
  );
}

module.exports = function catalogueRoutes({siteDir}) {
  const source = path.join(siteDir, 'src/data/catalogue.json');
  const repoRoot = path.resolve(siteDir, '..');
  return {
    name: 'clio-catalogue-routes',
    getPathsToWatch: () => [source],
    loadContent: async () => JSON.parse(await fs.readFile(source, 'utf8')),
    async contentLoaded({content, actions}) {
      const repositories = new Map(
        content.publishers.map((p) => [p.id, p.repository]),
      );
      const pins = await communityEntries(repoRoot);
      for (const item of content.items) {
        const modules = {
          entry: await actions.createData(
            `${item.id.replace('/', '-')}.json`,
            JSON.stringify(item),
          ),
        };
        if (item.installation === 'launcher') {
          modules.reference = path.resolve(
            siteDir,
            '..',
            `${item.docs.slice(1)}.md`,
          );
        }
        const local = item.path && path.resolve(repoRoot, item.path);
        const sourceFilePath =
          local && (await exists(local))
            ? local
            : pins.get(repositories.get(item.publisher));
        actions.addRoute({
          path: `/catalogue/${item.id}`,
          exact: true,
          component: '@site/src/components/Marketplace/CatalogueEntry.js',
          modules,
          ...(sourceFilePath && {metadata: {sourceFilePath}}),
        });
      }
    },
  };
};
