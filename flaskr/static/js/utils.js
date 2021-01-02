// Remove previously generated output
fetch(`${window.origin}/checks`, {
    method: "POST",
    credentials: "include",
    body: JSON.stringify({
        "remove": "remove"
    }),
    cache: "no-cache",
    headers: new Headers({
        "content-type": "application/json"
    })
});

function showLoader() {
    var loader = document.createElement("div");
    loader.className = "loader";
    document.getElementById("outputID").appendChild(loader);
}

function hideLoader() {
    document.getElementById("outputID").innerHTML = '';
}

function displayOutputImage() {
    var outputImg = document.createElement("img");
    outputImg.src = "/static/imgs/processed.png";
    outputImg.style = "max-width:100%; max-height:35rem;";
    outputImg.alt = "Output.png";
    // console.log("outputImg:", outputImg);
    download_link = `
        <a class="btn btn-info mb-3" href="/static/imgs/processed.png" download">
            <b>download</b>
        </a>`
    document.getElementById("outputID").innerHTML = download_link;
    document.getElementById("outputID").appendChild(outputImg);
}

function startProcessing() {
    showLoader();

    var entry = {
        "runButton": "clicked",
    };

    fetch(`${window.origin}/output`, {
        method: "POST",
        credentials: "include",
        body: JSON.stringify(entry),
        cache: "no-cache",
        headers: new Headers({
            "content-type": "application/json"
        })
    }).then(function(response) {
        if (response.status !== 200) {
            console.log("Error! Response not 200.")
        }

        response.json().then(function(data) {
            // console.log(data);
            if (data.num_images < 2) {
                hideLoader()
                alert("At least 2 images are required.")
            } else if (data.processing_complete === true) {
                displayOutputImage()
                    // alert("Processing complete")
            }
        });
    });
}