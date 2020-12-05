function create_dom(html_string) {
  var div = document.createElement('div');
  div.innerHTML = html_string.trim();

  // Change this to div.childNodes to support multiple top-level nodes
  return div.firstChild; 
}
function loading() {
  let dom = document.querySelector(".loading");
  if (dom.classList.contains("hidden")) {
    dom.classList.remove("hidden");
  }
}
function load_done() {
  let dom = document.querySelector(".loading");
  if (!dom.classList.contains("hidden")) {
    dom.classList.add("hidden");
  }
}

function show_result() {
  let dom = document.querySelector("#question");
  if (dom.classList.contains("focus")) {
    dom.classList.remove("focus");
  }
}

function hide_result() {
  let dom = document.querySelector("#question");
  if (!dom.classList.contains("focus")) {
    dom.classList.add("focus");
  }
}

function update_sparql(sparql) {
  let dom = document.querySelector("#sparql .content pre");
  dom.innerHTML = sparql;
}

function clear_answers() {
  let dom = document.querySelector("#answer #results");
  dom.innerHTML = "";
}

function add_answer(data) {
  let dom = document.querySelector("#answer #results");
  let answer_dom = `
    <div class="box">
      <div class="header">
        {{title}}
        <a target="_blank" href="{{wikipedia}}">WIKIPEDIA</a>
      </div>
      <div class="content">
        <span class="image">
          <img src="{{img_src}}" alt="{{title}}"">
        </span>
        <p class="type">
          <strong>Type:</strong>
          <span>{{type}}</span>
        </p>
        <p class="summary">{{summary}}</p>
      </div>
    </div> 
  `;
  answer_dom = answer_dom.replace(/{{title}}/g, data["title"])
                         .replace(/{{wikipedia}}/g, data["sitelink"] || "")
                         .replace(/{{img_src}}/g, data["image"] || "")
                         .replace(/{{summary}}/g, data["summary"] || "");

  if (data["type"]) answer_dom = answer_dom.replace(/{{type}}/g, data["types"].join(", ") || "");

  dom.appendChild(create_dom(answer_dom));
  console.log(data, data["image"])

  if (!data["sitelink"] || data["sitelink"] == undefined) {
    dom.querySelector(".box .header a").classList.add("hidden");
  }
  if (!data["image"] || data["image"] == undefined) {
    dom.querySelector(".box .content .image").classList.add("hidden");
    dom.querySelector(".box .content .image img").classList.add("hidden");
  }
  if (!data["type"] || data["type"] == undefined) {
    dom.querySelector(".box .content .type").classList.add("hidden");
  }
}

function quick_answer(answer) {
  Swal.fire({
    title: "ANSWER",
    html: answer,
    icon: "info"
  });
}

function handle_response(response) {
  if (response.sparql) update_sparql(response.sparql);
  if (response.status != 0) {
    Swal.fire({
      title: "ERROR",
      text: response.msg,
      icon: "error"
    });
  }
  if (response.values && response.values.length > 0) {
    response.values.forEach(value => {
      add_answer({
        title: value,
        summary: value
      });
    });
  }
  if (response.answers && response.answers.length > 0) {
    response.answers.forEach(answer => {
      WIKIPEDIA.getData(answer.sitelink, result => {
        result.summary.sitelink = answer.sitelink;
        add_answer(result.summary);
      }, error => {
        load_done();
        console.log(error)
      });
    });
  }
  if (!response.values && !response.answers) {
    Swal.fire({
      title: "Message",
      text: response.msg,
      icon: "info"
    });
  }
  show_result();
  load_done();
}

let form = document.querySelector("#question");
form.onsubmit = function(event) {
  event.preventDefault();

  let form_data = new FormData(form);

  if (form_data.get("query").length > 3) {
    clear_answers();
    hide_result();
    loading();

    fetch("/ask", {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({question: form_data.get("query")})
    })
    .then(response => {
      response.json().then(result => {
        console.log(result)
        handle_response(result);
      })
      .catch(error => {
        console.log(error)
        load_done();
      });
    })
    .catch(error => {
      console.log(error);
      load_done();
    });
  }
};
