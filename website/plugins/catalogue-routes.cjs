const fs = require('node:fs/promises');
const path = require('node:path');

module.exports = function catalogueRoutes({siteDir}) {
  const source = path.join(siteDir, 'src/data/catalogue.json');
  return {
    name: 'clio-catalogue-routes',
    getPathsToWatch: () => [source],
    loadContent: async () => JSON.parse(await fs.readFile(source, 'utf8')),
    async contentLoaded({content, actions}) {
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
        actions.addRoute({
          path: `/catalogue/${item.id}`,
          exact: true,
          component: '@site/src/components/Marketplace/CatalogueEntry.js',
          modules,
        });
      }
    },
  };
};
