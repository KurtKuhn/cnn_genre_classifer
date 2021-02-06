
// preventing the browser window from opening dropped files
window.addEventListener("dragover",function(e){
    e.preventDefault();
  },false);

window.addEventListener("drop",function(e){
    e.preventDefault();
  },false);
//


// drag and drop

// dz_container is for styling
let dz_container = document.querySelector('.dz_container');

// dropzone has for event listener
let dropzone = document.querySelector('.dropzone');

// adding dropzone event listeners
dropzone.addEventListener("dragenter", dragenter, false);
dropzone.addEventListener("dragover", dragover, false);
dropzone.addEventListener("dragleave", dragleave, false);
dropzone.addEventListener("drop", drop, false);

// when file dragged into dropzone
function dragenter(e) {
    e.stopPropagation()
    e.preventDefault()
    //highlight
    dropzone_bg.style.backgroundColor = "rgb(133, 133, 133, 1)";
}

// just stop default behavior and stop propogation here
function dragover(e) {
    e.stopPropagation()
    e.preventDefault() 
}

// when file is dragged out of dropzone
function dragleave(e) {
    e.stopPropagation()
    e.preventDefault()
    // remove highlight
    dropzone_bg.style.backgroundColor = "rgb(133, 133, 133, .8)";
}

// if file dropped in dropzone
//TODO: File validation
function drop(e) {
    e.stopPropagation();
    e.preventDefault();

    //gets file and appends it to FormData() object
    const dt = e.dataTransfer;
    const files = dt.files;
    const formData = new FormData() 
    formData.append('audioFile', files[0])

    // uses fetch to POST file to URL
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        
        // uncomment to see returned data
        //console.log(data)

        // CHART for CanvasJS
        var chart = new CanvasJS.Chart("chartContainer", {
            animationEnabled: true,
            theme: "dark1", // "light1", "light2", "dark1", "dark2"
            title:{
                text: "Genre Confidence"
            },
            axisY: {
                title: "Confidence"
            },
            data: [{        
                type: "column",  
                showInLegend: true, 
                legendMarkerColor: "grey",
                legendText: "Genres",
                dataPoints: [      
                    { y: data['predictions'][0][0], label: "Jazz" },
                    { y: data['predictions'][0][1],  label: "Reggae" },
                    { y: data['predictions'][0][2],  label: "Rock" },
                    { y: data['predictions'][0][3],  label: "Blues" },
                    { y: data['predictions'][0][4],  label: "HipHop" },
                    { y: data['predictions'][0][5], label: "Country" },
                    { y: data['predictions'][0][6],  label: "Metal" },
                    { y: data['predictions'][0][7],  label: "Classical" },
                    { y: data['predictions'][0][8],  label: "Disco" },
                    { y: data['predictions'][0][9],  label: "Pop" },
                ]
            }]
        });
        chart.render();
        
    })
    // fetch error
    .catch(error => {
        console.error(error)
    })
}
