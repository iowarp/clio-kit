const fs = require('node:fs/promises');
const path = require('node:path');

// Index for Clio Coder's "Find a guide": each doc's title, summary, sidebar
// group and section headings, published as global data for the sidebar search.
module.exports = function docSearch({siteDir}) {
  return {
    name: 'clio-doc-search',
    async allContentLoaded({allContent, actions}) {
      const [version] =
        allContent['docusaurus-plugin-content-docs'].default.loadedVersions;
      const groups = {};
      const walk = (items, group) => {
        for (const item of items) {
          if (item.type === 'category') walk(item.items, item.label);
          else if (item.type === 'doc') groups[item.id] = group;
        }
      };
      for (const items of Object.values(version.sidebars)) walk(items, 'Docs');
      const docs = version.docs.filter((doc) => !doc.unlisted && !doc.draft);
      actions.setGlobalData(
        await Promise.all(
          docs.map(async (doc) => {
            const source = await fs.readFile(
              path.join(siteDir, doc.source.replace('@site/', '')),
              'utf8',
            );
            const headings = [
              ...source
                .replace(/^```[\s\S]*?^```/gm, '')
                .matchAll(/^##\s+(.+)$/gm),
            ].map(([, text]) =>
              text
                .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
                .replace(/^\d+\.\s*|[`*]/g, '')
                .trim(),
            );
            return {
              title: doc.title,
              url: doc.permalink,
              group: groups[doc.id] ?? 'Docs',
              excerpt: doc.description,
              headings,
            };
          }),
        ),
      );
    },
  };
};
