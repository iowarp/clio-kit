import React from 'react';
import Link from '@docusaurus/Link';
import {Frame} from '../components/Marketplace/shared';
import {Catalog} from '../components/Marketplace/Catalog';
import styles from '../components/Marketplace/styles.module.css';

export default function CataloguePage() {
  return (
    <Frame
      title="Catalogue of scientific tools and workflows"
      description="Browse scientific MCP servers, skills, workflow plugins, agents and hooks for HPC, scientific data formats and research, with install steps for each client."
    >
      <div className={styles.container}>
        <header className={styles.catalogueIntro}>
          <p className={styles.kicker}>CLIO Kit / Meta-marketplace</p>
          <h1 className={styles.catalogueTitle}>Find your scientific toolkit.</h1>
          <p className={styles.lede}>
            One catalogue for CLIO Kit’s own components and the plugins it
            indexes from other publishers. Start with a workflow plugin, which
            bundles the MCP servers and skills a task needs, or choose
            components one at a time.
          </p>
          <Link to="/publishers" className={styles.textLink}>
            Meet the publishers →
          </Link>
        </header>
        <Catalog />
      </div>
    </Frame>
  );
}
