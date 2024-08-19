document.addEventListener('DOMContentLoaded', function() {
  const container = document.querySelector('.container');
  const fileInput = document.getElementsByClassName('form-file')[0];
  const formContainer = document.getElementsByClassName('form')[0];
  const loaderContainer = document.querySelector('.loader-container');

  function handlePaste(event) {
    const items = event.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith('image/')) {
        const file = items[i].getAsFile();
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        break;
      }
    }
  }

  formContainer.addEventListener('paste', handlePaste);

  const closeButton = document.getElementById('close-button');
  closeButton.addEventListener('click', () => {
    window.close();
  });

  const submitButton = document.getElementById('form-submit');

  submitButton.addEventListener('click', () => {
    container.style.display = 'none';
    loaderContainer.classList.remove('hidden');

    setTimeout(() => {
      loaderContainer.classList.add('hidden');
      container.style.display = 'flex';
    }, 2000);
  }); 
});