// Open all external links in a new tab
// source: https://stackoverflow.com/a/62742435/4063376
$(document).ready(function () {
   $('a[href^="http://"], a[href^="https://"]').not('a[class*=internal]').attr('target', '_blank');
});
