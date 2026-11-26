/** @type {import('next').NextConfig} */
const nextConfig = {
  // The widget is meant to be embedded in another shop's page, so it
  // must be servable from any origin. The API (not this app) owns CORS.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [{ key: "X-Frame-Options", value: "ALLOWALL" }],
      },
    ];
  },
};

module.exports = nextConfig;
