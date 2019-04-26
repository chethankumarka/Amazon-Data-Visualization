// DATA
var words_product = {"hard": {"polarity": -0.2916666666666667, "count": 11}, "great": {"polarity": 0.8, "count": 31}, "wrong": {"polarity": -0.5, "count": 4}, "beautiful": {"polarity": 0.85, "count": 14}, "expensive": {"polarity": -0.5, "count": 4}, "good": {"polarity": 0.7, "count": 14}, "stupid": {"polarity": -0.7999999999999999, "count": 2}, "sound": {"polarity": 0.4, "count": 23}, "fake": {"polarity": -0.5, "count": 3}, "nice": {"polarity": 0.6, "count": 13}, "bad": {"polarity": -0.6999999999999998, "count": 2}, "better": {"polarity": 0.5, "count": 14}, "horrible": {"polarity": -1.0, "count": 1}, "perfect": {"polarity": 1.0, "count": 7}, "little": {"polarity": -0.1875, "count": 5}, "best": {"polarity": 1.0, "count": 6}, "black": {"polarity": -0.16666666666666666, "count": 5}, "amazing": {"polarity": 0.6000000000000001, "count": 7}, "stupidly": {"polarity": -0.7999999999999999, "count": 1}, "love": {"polarity": 0.5, "count": 8}, "disappointed": {"polarity": -0.75, "count": 1}, "happy": {"polarity": 0.8, "count": 4}, "sick": {"polarity": -0.7142857142857143, "count": 1}, "sure": {"polarity": 0.5, "count": 6}, "expected": {"polarity": -0.1, "count": 6}, "really": {"polarity": 0.2, "count": 13}, "blasted": {"polarity": -0.6, "count": 1}, "easy": {"polarity": 0.43333333333333335, "count": 5}, "twisted": {"polarity": -0.5, "count": 1}, "bright": {"polarity": 0.7000000000000001, "count": 3}, "failed": {"polarity": -0.5, "count": 1}, "impressive": {"polarity": 1.0, "count": 2}, "small": {"polarity": -0.25, "count": 2}, "superb": {"polarity": 1.0, "count": 2}, "dusty": {"polarity": -0.4, "count": 1}, "top": {"polarity": 0.5, "count": 4}, "annoyed": {"polarity": -0.4, "count": 1}, "many": {"polarity": 0.5, "count": 4}, "thin": {"polarity": -0.4, "count": 1}, "perfectly": {"polarity": 1.0, "count": 2}, "dumb": {"polarity": -0.375, "count": 1}, "awesome": {"polarity": 1.0, "count": 2}, "sucks": {"polarity": -0.3, "count": 1}, "excellent": {"polarity": 1.0, "count": 2}, "least": {"polarity": -0.3, "count": 1}, "beautifully": {"polarity": 0.85, "count": 2}, "careful": {"polarity": -0.1, "count": 2}, "free": {"polarity": 0.4, "count": 4}, "slightly": {"polarity": -0.16666666666666666, "count": 1}, "pretty": {"polarity": 0.25, "count": 6}, "less": {"polarity": -0.16666666666666666, "count": 1}, "gorgeous": {"polarity": 0.7, "count": 2}, "half": {"polarity": -0.16666666666666666, "count": 1}, "loved": {"polarity": 0.7, "count": 2}, "previously": {"polarity": -0.16666666666666666, "count": 1}, "new": {"polarity": 0.13636363636363635, "count": 10}, "bass": {"polarity": -0.15000000000000002, "count": 1}, "warm": {"polarity": 0.6, "count": 2}, "dark": {"polarity": -0.15, "count": 1}, "honestly": {"polarity": 0.6, "count": 2}, "sharp": {"polarity": -0.125, "count": 1}, "nicely": {"polarity": 0.6, "count": 2}, "extremely": {"polarity": -0.125, "count": 1}, "impressed": {"polarity": 1.0, "count": 1}, "wide": {"polarity": -0.1, "count": 1}, "authentic": {"polarity": 0.5, "count": 2}, "center": {"polarity": -0.1, "count": 1}, "outstanding": {"polarity": 0.5, "count": 2}, "long": {"polarity": -0.05, "count": 2}, "flawlessly": {"polarity": 1.0, "count": 1}, "rough": {"polarity": -0.1, "count": 1}, "much": {"polarity": 0.2, "count": 5}, "came": {"polarity": 0.0, "count": 13}, "first": {"polarity": 0.25, "count": 4}, "condition": {"polarity": 0.0, "count": 2}, "safe": {"polarity": 0.5, "count": 2}, "time": {"polarity": 0.0, "count": 7}, "incredibly": {"polarity": 0.9, "count": 1}, "even": {"polarity": 0.0, "count": 5}, "fantastic": {"polarity": 0.4, "count": 2}, "case": {"polarity": 0.0, "count": 30}, "experienced": {"polarity": 0.8, "count": 1}, "locks": {"polarity": 0.0, "count": 1}, "enjoy": {"polarity": 0.4, "count": 2}, "use": {"polarity": 0.0, "count": 1}, "cheap": {"polarity": 0.4, "count": 2}, "boyfriend": {"polarity": 0.0, "count": 2}, "absolutely": {"polarity": 0.2, "count": 4}, "loves": {"polarity": 0.0, "count": 1}, "rich": {"polarity": 0.375, "count": 2}, "uses": {"polarity": 0.0, "count": 1}, "interested": {"polarity": 0.25, "count": 3}, "play": {"polarity": 0.0, "count": 10}, "wise": {"polarity": 0.7, "count": 1}, "music": {"polarity": 0.0, "count": 6}, "exceptional": {"polarity": 0.6666666666666666, "count": 1}, "product": {"polarity": 0.0, "count": 6}, "super": {"polarity": 0.3333333333333333, "count": 2}}
var words_london = {}
var shuffling = true;				// shuffling make less-sparsed word clouds
var spiral = "archimedean";		// 'archimedean or 'rectangular'
var outter = true;						// (try to) make disappearing words outter

var city = "product";
var zoom_group;							// reusable d3 selection

var max = -Infinity;
var font_size = d3.scale.linear()
.domain([1, max])
.range([10,100]);

var color = d3.scale.quantize()
.domain([-max, max])
.range([d3.rgb(203, 130, 116), d3.rgb(122, 203, 116)]);

var updateWordCloud = function(varName,wordCloudKey)
{
	console.log("in updateWordCloud"+varName+":::"+wordCloudKey);
	var words = {};
	words_product = window[varName][wordCloudKey];
	d3.select("#column1").selectAll('*').remove();
Object.keys(words_product).forEach(function(word_lemma){
  words[word_lemma] = {
    "product": words_product[word_lemma],
    "london": {"polarity":words_product[word_lemma].polarity,"count":0}
  }
});

Object.keys(words_london).forEach(function(word_lemma){
  if (words[word_lemma]) {
    words[word_lemma].london = words_london[word_lemma];
  } else {
    w = words[word_lemma] = {
      "product": {"polarity":words_london[word_lemma].polarity,"count":0},
      "london": words_london[word_lemma]
    }
  }
});

Math.seedrandom('10');  // define a fixed random seed, to avoid to have a different layout on each page reload. change the string to randomize

var words_frequency = [];

Object.keys(words).forEach(function(word_lemma){
  o = words[word_lemma];
  o.lemma = word_lemma;
  o.maxCount = Math.max(o.product.count, o.london.count);
  
  words_frequency.push(o);
  if (max < o.maxCount) {
    max = o.maxCount;
  }
});
console.log("max: "+max);

font_size = d3.scale.linear()
.domain([1, max])
.range([10,100]);

color = d3.scale.quantize()
.domain([-max, max])
.range([d3.rgb(203, 130, 116), d3.rgb(122, 203, 116)]);


/*
var color = d3.scale.linear()
    .domain([-max, 0, max])
    .range([d3.hcl(36, 65, 50), d3.hcl(95, 65, 80), d3.hcl(150, 65, 50)])
    .interpolate(d3.interpolateHcl);*/



if (shuffling) {
  shuffle(words_frequency);
}
words_frequency.sort(function(a,b){
  if (outter) {
  	return (b.maxCount - a.maxCount)
  					+ (((b.product.count === 0) || (b.london.count === 0))? 1 : 0);
  } else {
    return (b.maxCount - a.maxCount);
  }
});

d3.layout.cloud().size([400, 400])
  .words(words_frequency)
  .rotate(function() { return ~~(Math.random() * 2) * 90; })
  .font("Impact")
  .spiral(spiral)
  .text(function(d){ return d.lemma; })
  .fontSize(function(d){ return font_size(d.maxCount); })
  .on("end", draw)
  .start();
}


function draw(words) {
  /*var svg = d3.select("body").append("svg")
  .attr("width", 960)
  .attr("height", 500);*/
  
  var svg = d3.select("#column1").append("svg")
  .attr("width", 500)
  .attr("height", 430);


  // append a group for zoomable content
  zoom_group = svg.append('g');

  // define a zoom behavior
  var zoom = d3.behavior.zoom()
  .scaleExtent([1,4]) // min-max zoom
  .on('zoom', function() {
    // whenever the user zooms,
    // modify translation and scale of the zoom group accordingly
    zoom_group.attr('transform', 'translate('+zoom.translate()+')scale('+zoom.scale()+')');
  });

  // bind the zoom behavior to the main SVG
  svg.call(zoom);


  zoom_group.append("g")
    .attr("transform", "translate(280,250)")
    .selectAll("text")
    .data(words)
    .enter()
    	.append("text")
      .style("font-size", function(d){ return font_size(d[city].count) + "px"; })
      .style("font-family", "Impact")
      .style("fill", function(d) { return color(d[city].polarity); })
  		.style("fill-opacity", function(d) { return (d[city].count>0)? 1 : 0; })
      .attr("text-anchor", "middle")
      .attr("transform", function(d) {
        var far = 1500*(Math.random() > 0.5 ? +1 : -1);
        if(d.rotate === 0)
          return "translate("+far+",0)rotate(" + d.rotate + ")";
        else
          return "translate(0,"+far+")rotate(" + d.rotate + ")";
      })
      .text(function(d) { return d.lemma; })
      .transition()
        .duration(1000)
        .attr("transform", function(d) {
          return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
        });
	
	d3.select("#column1").append("span").attr("transform", "translate(-50,10)");
}

function update() {
  if (city === "london") {
    city = "product";
  } else {
    city = "london";
  }
  
  zoom_group.selectAll("text")
  	.data(words_frequency)
  	.transition()
      .duration(1000)
      .style("font-size", function(d){ return font_size(d[city].count) + "px"; })
  		.style("fill", function(d) { return color(d[city].polarity); })
  		.style("fill-opacity", function(d) { return (d[city].count>0)? 1 : 0; });
}

function shuffle(array) {
  let counter = array.length;

  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    let index = Math.floor(Math.random() * counter);

    // Decrease counter by 1
    counter--;

    // And swap the last element with it
    let temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }

  return array;
}