$(document).ready(() => {
    // scroll to bottom if prediction is there
    if ($('#prediction').length) {
        console.log('it here', $(document).height());
        $('html').animate({ scrollTop: $(document).height() }, 'slow');
        console.log('done');
    }
})