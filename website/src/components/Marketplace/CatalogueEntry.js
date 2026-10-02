import React from 'react';
import MDXContent from '@theme/MDXContent';
import {Frame} from './shared';
import {DetailContent} from './Details';
import styles from './styles.module.css';

export default function CatalogueEntry({entry, reference: Reference}) {
  return (
    <Frame title={entry.title}>
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
