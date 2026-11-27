import React from 'react';
import s from './overview.module.css';

export default function LogoMark({hero = false}) {
  return (
    <div className={s.logoScene} data-logo-slot aria-hidden="true">
      <img
        src="/img/iowarp_logo.png"
        width="1080"
        height="1080"
        alt=""
        loading={hero ? 'eager' : 'lazy'}
        decoding="async"
      />
    </div>
  );
}
