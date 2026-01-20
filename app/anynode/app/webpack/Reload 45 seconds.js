setInterval(function() {
   // Scrolls to a specific position on the page
   window.scrollTo({
      top: 1280, // Change '100' to the vertical pixel position you want to scroll to
      behavior: 'smooth' // Smooth scrolling
   });

   // Refreshes the page
   location.reload();
}, 15000); // 45 seconds interval
