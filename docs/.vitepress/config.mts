import { defineConfig } from 'vitepress';

export default defineConfig({
  lang: 'zh-CN',
  title: 'SPIDER for h2r_bench',
  description: 'SPIDER on benchmark-owned tasks and assets',
  base: '/spider/',
  themeConfig: {
    nav: [
      { text: 'H2R Usage', link: '/h2r' },
      { text: 'Upstream', link: 'https://github.com/facebookresearch/spider' },
    ],
    sidebar: [
      { text: 'H2R Usage', link: '/h2r' },
      { text: 'Upstream Differences', link: '/upstream-differences' },
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/dex-synth/spider' }],
  },
});
