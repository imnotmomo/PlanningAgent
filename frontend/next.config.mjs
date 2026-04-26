/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow the Cloudflare tunnel host so the Next.js dev server's CORS check
  // doesn't refuse requests proxied through trycloudflare.
  allowedDevOrigins: [
    "guided-inexpensive-partnerships-filing.trycloudflare.com",
  ],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
