// quantfin-exeter edge entry.
//
// The site is served from quantfin.dev/execution via a service binding on
// the landing worker, so requests that still arrive on the old
// exeter.quantfin.dev host are redirected to the canonical URL.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.hostname === "exeter.quantfin.dev") {
      const target = url.pathname === "/" ? "" : url.pathname;
      return Response.redirect(`https://quantfin.dev/execution${target}${url.search}`, 301);
    }

    return env.ASSETS.fetch(request);
  },
};
