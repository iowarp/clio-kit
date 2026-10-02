import React from 'react';
import {Frame} from '../components/Marketplace/shared';
import {Catalog} from '../components/Marketplace/Catalog';
import styles from '../components/Marketplace/styles.module.css';

export default function CataloguePage() {
  return (
    <Frame title="Scientific tools & workflows — Catalogue">
      <div className={styles.container}>
        <h1 className={styles.catalogueTitle}>Find your scientific toolkit.</h1>
        <Catalog />
      </div>
    </Frame>
  );
}
