// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'CLIO Kit',
  titleDelimiter: '—',
  tagline: 'A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions. Part of the IOWarp platform.',
  favicon: 'img/iowarp_logo.png',

  // Site-wide structured data; pages add their own entities through <Head>.
  headTags: [
    {
      tagName: 'script',
      attributes: {type: 'application/ld+json'},
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@graph': [
          {
            '@type': 'WebSite',
            '@id': 'https://toolkit.iowarp.ai/#site',
            name: 'CLIO Kit',
            url: 'https://toolkit.iowarp.ai/',
            publisher: {'@id': 'https://toolkit.iowarp.ai/#org'},
            inLanguage: 'en',
          },
          {
            '@type': 'Organization',
            '@id': 'https://toolkit.iowarp.ai/#org',
            name: 'Gnosis Research Center',
            url: 'https://grc.iit.edu/',
            logo: 'https://toolkit.iowarp.ai/img/iowarp_logo.png',
            parentOrganization: {
              '@type': 'CollegeOrUniversity',
              name: 'Illinois Institute of Technology',
            },
          },
        ],
      }),
    },
    {
      tagName: 'link',
      attributes: {rel: 'apple-touch-icon', href: '/img/iowarp_logo.png'},
    },
  ],

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

  plugins: ['./plugins/catalogue-routes.cjs', './plugins/doc-search.cjs'],

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
        blog: false,
        sitemap: {ignorePatterns: ['/component']},
        theme: {
          customCss: ['./src/css/custom.css', './src/css/editorial.css'],
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Titles, descriptions and og:url are set per page; these are fallbacks.
      metadata: [
        {name: 'description', content: 'CLIO Kit is a meta-marketplace for scientific MCP servers, skills, workflow plugins, agents and hooks. Part of the IOWarp platform.'},
        {name: 'theme-color', content: '#000000'},
        {name: 'color-scheme', content: 'dark light'},
        {property: 'og:site_name', content: 'CLIO Kit'},
        {name: 'twitter:card', content: 'summary_large_image'},
      ],
      // Social card for link previews; source is website/social-card.html.
      image: 'img/social-card.png',
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
          // Install steps live on the overview, so the header button opens the demos.
          {
            to: '/demos',
            position: 'right',
            label: 'Demos',
            className: 'kit-cta-link',
          },
          {
            href: 'https://github.com/iowarp/clio-kit',
            label: 'GitHub ↗',
            position: 'right',
            className: 'kit-side-link',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {label: 'Docs', to: '/docs/intro'},
          {label: 'Tutorials', to: '/tutorials'},
          {label: 'Demos', to: '/demos'},
          {label: 'Catalogue', to: '/catalogue'},
          {label: 'GitHub', href: 'https://github.com/iowarp/clio-kit'},
          {label: 'BSD-3-Clause', href: 'https://github.com/iowarp/clio-kit/blob/main/LICENSE'},
          {label: 'Gnosis Research Center', href: 'https://grc.iit.edu/'},
        ],
        copyright: `Copyright ${new Date().getFullYear()} iowarp.ai`,
      },
      prism: {
        // Code blocks are dark navy in both themes.
        theme: prismThemes.nightOwl,
        darkTheme: prismThemes.nightOwl,
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
