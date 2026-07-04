/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    HF_BACKEND_URL: 'https://sk-saniya-animal-vision-backend.hf.space',
    NEXT_PUBLIC_HF_BACKEND_URL: 'https://sk-saniya-animal-vision-backend.hf.space',
  },
};

module.exports = nextConfig;