import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Footer() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <footer className="kit-footer">
      <div className="kit-footer-inner">
        <div className="kit-footer-top">
          <Link className="kit-footer-brand" to="/">
            <img src="/img/iowarp_logo.png" width="32" height="32" alt="" />
            CLIO Kit
          </Link>
          <p>A meta-marketplace for scientific tools and knowledge.</p>
        </div>
        <div className="kit-footer-bottom">
          <span>{siteConfig.themeConfig.footer.copyright}</span>
          <nav aria-label="Footer">
            <a href="https://grc.iit.edu/">
              Developed by Gnosis Research Center
            </a>
            <a href="https://github.com/iowarp/clio-kit/blob/main/LICENSE">
              BSD-3-Clause
            </a>
            <a href="https://github.com/iowarp/clio-kit/issues">Feedback ↗</a>
          </nav>
        </div>
      </div>
    </footer>
  );
}
