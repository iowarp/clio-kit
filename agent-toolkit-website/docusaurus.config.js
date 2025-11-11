// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Agent Toolkit - Gnosis Research Center',
  tagline: 'Tools, skills, plugins, and extensions for AI agents. Part of the IoWarp platform. | Developed by Gnosis Research Center (GRC) at Illinois Institute of Technology',
  favicon: 'img/iowarp_logo.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://iowarp.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/agent-toolkit/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'iowarp', // Usually your GitHub org/user name.
  projectName: 'agent-toolkit', // Usually your repo name.

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
            'https://github.com/iowarp/agent-toolkit/tree/main/docs/',
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
        {name: 'description', content: 'Agent Toolkit v1.0.0 (Beta Public Release - November 11, 2025) - Tools, skills, plugins, and extensions for AI agents. Part of the IoWarp platform. Features 15+ MCP servers for scientific computing: HDF5, Slurm, Pandas, ArXiv, and more. Built with FastMCP 2.12, 150+ tools for HPC workflows. Developed by Gnosis Research Center at Illinois Institute of Technology, supported by NSF.'},
        {name: 'keywords', content: 'Agent Toolkit, AI agents, tools, skills, plugins, extensions, MCP, Model Context Protocol, scientific computing, HPC, HDF5, Slurm, Pandas, ADIOS, Parquet, FastMCP, research computing, IoWarp platform, Gnosis Research Center, Illinois Tech, NSF'},
        {property: 'og:title', content: 'Agent Toolkit - Tools for AI Agents | IoWarp Platform | Gnosis Research Center'},
        {property: 'og:description', content: 'Agent Toolkit v1.0.0 (Beta Public Release - November 11, 2025): 15+ MCP servers for scientific computing. Part of the IoWarp platform providing tools, skills, plugins, and extensions for AI agents. HDF5, Slurm, Pandas, ArXiv. Built with FastMCP 2.12 at Illinois Institute of Technology.'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'Agent Toolkit - Tools for AI Agents | IoWarp Platform'},
        {name: 'twitter:description', content: 'Agent Toolkit v1.0.0 (Beta Public Release - November 11, 2025): MCP servers for scientific computing. Part of the IoWarp platform providing comprehensive agent tooling.'},
      ],
      // Social card for link previews
      image: 'img/iowarp_logo.png',
      navbar: {
        title: 'Agent Toolkit',
        logo: {
          alt: 'Agent Toolkit Logo',
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
            href: 'https://pypi.org/project/agent-toolkit/',
            position: 'right',
            label: 'PyPI',
          },
          {
            href: 'https://grc.iit.edu/',
            position: 'right',
            label: 'GRC',
          },
          {
            href: 'https://github.com/iowarp/agent-toolkit',
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
            title: 'Agent Toolkit',
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
                label: 'Platform Website',
                href: 'https://iowarp.ai',
              },
              {
                label: 'GitHub Repository',
                href: 'https://github.com/iowarp/agent-toolkit',
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
                href: 'https://github.com/iowarp/agent-toolkit',
              },
              {
                label: 'Issue Tracker',
                href: 'https://github.com/iowarp/agent-toolkit/issues',
              },
              {
                label: 'Zulip Chat',
                href: 'https://grc.zulipchat.com/#narrow/channel/518574-agent-toolkit',
              },
            ],
          },
          {
            title: 'Distribution',
            items: [
              {
                label: 'PyPI Package',
                href: 'https://pypi.org/project/agent-toolkit/',
              },
              {
                label: 'Release Notes',
                href: 'https://github.com/iowarp/agent-toolkit/releases',
              },
            ],
          },
        ],
        copyright: `Agent Toolkit · Part of the IoWarp Platform · Developed by Gnosis Research Center (GRC), Illinois Institute of Technology. Funded in part by the National Science Foundation. © ${new Date().getFullYear()}`,
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
