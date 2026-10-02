// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'CLIO Kit - Gnosis Research Center',
  tagline: 'A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions. Part of the IoWarp platform.',
  favicon: 'img/iowarp_logo.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
    faster: false, // Keep the existing webpack build pipeline.
  },

  trailingSlash: false,

  // Set the production url of your site here
  url: 'https://toolkit.iowarp.ai',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'iowarp', // Usually your GitHub org/user name.
  projectName: 'clio-kit', // Usually your repo name.

  onBrokenLinks: 'throw',
  markdown: {hooks: {onBrokenMarkdownLinks: 'throw'}},

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  plugins: ['./plugins/catalogue-routes.cjs'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../docs',
          sidebarPath: './sidebars.js',
          routeBasePath: 'docs',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/iowarp/clio-kit/tree/main/docs/',
        },
        theme: {
          customCss: ['./src/css/custom.css', './src/css/editorial.css'],
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Enhanced metadata for social sharing
      metadata: [
        {name: 'description', content: 'CLIO Kit - A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions'},
        {name: 'keywords', content: 'CLIO Kit, meta-marketplace, AI agents, tools, skills, plugins, agents, community contributions, MCP, Model Context Protocol, scientific computing, HPC, HDF5, Slurm, Pandas, ADIOS, Parquet, FastMCP, research computing, IoWarp platform, Gnosis Research Center, Illinois Tech, NSF'},
        {property: 'og:title', content: 'CLIO Kit - Scientific AI Meta-Marketplace | IoWarp Platform | Gnosis Research Center'},
        {property: 'og:description', content: 'CLIO Kit - A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'CLIO Kit - Scientific AI Meta-Marketplace | IoWarp Platform'},
        {name: 'twitter:description', content: 'CLIO Kit - A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions'},
      ],
      // Social card for link previews
      image: 'img/iowarp_logo.png',
      navbar: {
        title: 'CLIO Kit',
        logo: {
          alt: 'CLIO Kit Logo',
          src: 'img/iowarp_logo.png',
        },
        items: [
          {
            to: '/',
            position: 'left',
            label: 'Overview',
            exact: true,
          },
          {
            to: '/catalogue',
            position: 'left',
            label: 'Catalogue',
            activeBaseRegex: '^/(catalogue|component|publishers)',
          },
          {
            to: '/docs',
            position: 'left',
            label: 'Docs',
            activeBaseRegex: '^/docs(?!/tutorials)',
          },
          {
            to: '/tutorials',
            position: 'left',
            label: 'Tutorials',
            activeBaseRegex: '^/(tutorials|docs/tutorials)',
          },
          {
            to: '/#start',
            position: 'right',
            label: 'Install CLIO Kit ↗',
            className: 'kit-install-link',
          },
          {
            href: 'https://github.com/iowarp/clio-kit',
            label: 'GitHub',
            position: 'right',
            className: 'navbar__icon-link navbar__icon-link--github',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {label: 'Docs', to: '/docs/intro'},
          {label: 'Tutorials', to: '/tutorials'},
          {label: 'Catalogue', to: '/catalogue'},
          {label: 'GitHub', href: 'https://github.com/iowarp/clio-kit'},
          {label: 'BSD-3-Clause', href: 'https://github.com/iowarp/clio-kit/blob/main/LICENSE'},
          {label: 'Gnosis Research Center', href: 'https://grc.iit.edu/'},
        ],
        copyright: `CLIO Kit · Part of the IoWarp Platform · © ${new Date().getFullYear()}`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
