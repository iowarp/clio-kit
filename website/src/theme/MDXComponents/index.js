import React from 'react';
import MDXComponents from '@theme-original/MDXComponents';

export default {
  ...MDXComponents,
  // Docusaurus tables scroll horizontally on small screens. Keep that region
  // reachable by keyboard even when the table has no links or other controls.
  table: (props) => <table tabIndex={0} {...props} />,
};
