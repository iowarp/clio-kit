import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Clio Coder's two-weight brand: the family name strong, the product name light.
export default function NavbarLogo() {
  return (
    <Link to="/" className="navbar__brand" aria-label="CLIO Kit home">
      <img
        className="navbar__logo"
        src={useBaseUrl('/img/iowarp_logo.png')}
        width="36"
        height="36"
        alt=""
      />
      <span className="kit-brand-name">
        CLIO <span>Kit</span>
      </span>
    </Link>
  );
}
