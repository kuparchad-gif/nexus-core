function togglePopup(event, show) {
    var popupBtn = event.target;
    var popupId = popupBtn.getAttribute('data-popup-id');
    var popup = document.getElementById(popupId);

    popup.parentElement.parentElement.parentElement.parentElement.parentElement.style.overflow = 'visible';
    
    popup.classList.toggle("show");
    popupBtn.parentElement.style.overflow = 'visible';

    event.stopPropagation();
}

document.addEventListener("mouseover", function(event) {
    if (event.target.classList.contains('popup-btn')) {
        togglePopup(event, true);
    }
});

document.addEventListener("mouseout", function(event) {
    if (event.target.classList.contains('popup-btn')) {
        togglePopup(event, false);
    }
});
