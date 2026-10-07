import React from 'react';
import Metadata from '@theme-original/DocItem/Metadata';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useDoc} from '@docusaurus/plugin-content-docs/client';

// Mirrors coder.iowarp.ai: guides are TechArticles, tutorials are Articles.
export default function MetadataWrapper(props) {
  const {metadata, frontMatter} = useDoc();
  const {siteConfig} = useDocusaurusContext();
  const url = `${siteConfig.url}${metadata.permalink}`;
  const article = {
    '@context': 'https://schema.org',
    '@type': metadata.id.startsWith('tutorials/') ? 'Article' : 'TechArticle',
    '@id': `${url}#article`,
    headline: metadata.title,
    description: metadata.description,
    url,
    mainEntityOfPage: url,
    isPartOf: {'@id': `${siteConfig.url}/#site`},
    inLanguage: 'en',
    image: `${siteConfig.url}${frontMatter.image || '/img/social-card.png'}`,
    author: {'@id': `${siteConfig.url}/#org`},
  };
  return (
    <>
      <Metadata {...props} />
      <Head>
        <meta property="og:type" content="article" />
        <script type="application/ld+json">{JSON.stringify(article)}</script>
      </Head>
    </>
  );
}
