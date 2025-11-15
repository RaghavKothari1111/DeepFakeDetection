/** @type {import('next').NextConfig} */
const { i18n } = require('./next-i18next.config');

const nextConfig = {
  reactStrictMode: true,
  i18n,
  // remove experimental.appDir if present for your Next version,
  // or update Next to a version that supports appDir.
};

module.exports = nextConfig;
