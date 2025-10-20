// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'IoWarp MCPs - Gnosis Research Center',
  tagline: 'Model Context Protocol servers for scientific computing research | Gnosis Research Center, Illinois Institute of Technology',
  favicon: 'img/iowarp_logo.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://iowarp.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/iowarp-mcps/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'iowarp', // Usually your GitHub org/user name.
  projectName: 'iowarp-mcps', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: false,
          routeBasePath: 'docs',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/iowarp/iowarp-mcps/tree/main/docs/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Enhanced metadata for social sharing
      metadata: [
        {name: 'description', content: '15 Model Context Protocol servers for scientific computing: HDF5, Slurm, Pandas, ArXiv, and more. Built with FastMCP 2.12, 150+ tools for HPC workflows. Developed by Gnosis Research Center at Illinois Institute of Technology, supported by NSF.'},
        {name: 'keywords', content: 'MCP, Model Context Protocol, scientific computing, HPC, HDF5, Slurm, Pandas, ADIOS, Parquet, FastMCP, AI agents, research computing, IoWarp, Gnosis Research Center, Illinois Tech, NSF'},
        {property: 'og:title', content: 'IoWarp MCPs - AI Tools for Scientific Computing | Gnosis Research Center'},
        {property: 'og:description', content: '15 MCP servers, 150+ tools for scientific computing. HDF5, Slurm, Pandas, ArXiv. Built with FastMCP 2.12 at Illinois Institute of Technology.'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'IoWarp MCPs - Gnosis Research Center'},
        {name: 'twitter:description', content: 'Scientific computing tools for AI integration via Model Context Protocol.'},
      ],
      // Social card for link previews
      image: 'img/iowarp_logo.png',
      navbar: {
        title: 'IoWarp MCPs',
        logo: {
          alt: 'IoWarp MCPs Logo',
          src: 'img/iowarp_logo.png',
        },
        items: [
          {
            to: '/',
            position: 'left',
            label: 'Browse MCPs',
          },
          {
            to: '/docs/intro',
            position: 'left',
            label: 'Getting Started',
          },
          {
            href: 'https://pypi.org/project/iowarp-mcps/',
            position: 'right',
            label: 'PyPI',
          },
          {
            href: 'https://grc.iit.edu/',
            position: 'right',
            label: 'GRC',
          },
          {
            href: 'https://github.com/iowarp/iowarp-mcps',
            label: 'GitHub',
            position: 'right',
            className: 'navbar__icon-link navbar__icon-link--github',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'IoWarp MCPs',
            items: [
              {
                label: 'Project Overview',
                to: '/docs/intro',
              },
              {
                label: 'Browse Servers',
                to: '/',
              },
              {
                label: 'Canonical Site',
                href: 'https://iowarp.github.io/iowarp-mcps/',
              },
            ],
          },
          {
            title: 'Research & Funding',
            items: [
              {
                label: 'National Science Foundation',
                href: 'https://new.nsf.gov/',
              },
              {
                label: 'Gnosis Research Center',
                href: 'https://grc.iit.edu/',
              },
              {
                label: 'Illinois Tech',
                href: 'https://www.iit.edu/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Repository',
                href: 'https://github.com/iowarp/iowarp-mcps',
              },
              {
                label: 'Issue Tracker',
                href: 'https://github.com/iowarp/iowarp-mcps/issues',
              },
              {
                label: 'Zulip Chat',
                href: 'https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps',
              },
            ],
          },
          {
            title: 'Distribution',
            items: [
              {
                label: 'PyPI Package',
                href: 'https://pypi.org/project/iowarp-mcps/',
              },
              {
                label: 'Release Notes',
                href: 'https://github.com/iowarp/iowarp-mcps/releases',
              },
            ],
          },
        ],
        copyright: `IoWarp MCPs · Gnosis Research Center, Illinois Institute of Technology. Funded in part by the National Science Foundation. © ${new Date().getFullYear()}`,
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
