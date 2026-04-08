/**
 * Run page interaction helpers for the Execution Lab.
 * HTMX handles the AJAX requests; this file provides minor UI enhancements.
 */

document.addEventListener('DOMContentLoaded', function () {
  // Show a top progress bar during any HTMX request
  document.body.addEventListener('htmx:beforeRequest', function () {
    document.getElementById('results-container')?.classList.add('opacity-60');
  });
  document.body.addEventListener('htmx:afterRequest', function () {
    document.getElementById('results-container')?.classList.remove('opacity-60');
  });

  // Scroll to results after full pipeline completes
  document.body.addEventListener('htmx:afterSettle', function (evt) {
    if (evt.detail && evt.detail.target && evt.detail.target.id === 'results-container') {
      evt.detail.target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
