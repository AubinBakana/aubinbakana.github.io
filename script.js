// Toggles visibility of CS50 deadline object
function blink()
{
    let body = document.querySelector('#wellcome');
    if (body.style.visibility == 'hidden')
    {
        body.style.visibility = 'visible';
    }
    else
    {
        body.style.visibility = 'hidden';
    }
}

// Blink every 33100ms
window.setInterval(blink, 1240);
