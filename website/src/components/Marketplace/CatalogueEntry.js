import React from 'react';
import MDXContent from '@theme/MDXContent';
import {Frame, metaDescription} from './shared';
import {DetailContent} from './Details';
import {kindName} from './data';
import styles from './styles.module.css';


export default function CatalogueEntry({entry, reference: Reference}) {
  const description = metaDescription(entry.summary);
  const keywords = [
    ...new Map(entry.tags.map((tag) => [tag.toLowerCase(), tag])).values(),
  ];
  const code = {
    '@type': 'SoftwareSourceCode',
    name: entry.title,
    description,
    codeRepository: entry.source,
    license: entry.license || undefined,
    version: entry.version || undefined,
    keywords: keywords.length ? keywords.join(', ') : undefined,
  };
  return (
    <Frame
      title={`${entry.title} · ${kindName(entry.kind)}`}
      description={description}
      crumbs={[{name: 'Catalogue', path: '/catalogue'}]}
      entities={[code]}
      // Launcher MCPs render their docs page verbatim; credit the original.
      canonical={Reference ? entry.docs : undefined}
    >
      <div className={styles.detailPage}>
        {Reference ? (
          <div className="markdown">
            <h1 className={styles.referenceTitle}>{entry.title}</h1>
            <MDXContent>
              <Reference />
            </MDXContent>
          </div>
        ) : (
          <DetailContent item={entry} />
        )}
      </div>
    </Frame>
  );
}
