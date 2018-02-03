// my changed file

var core = {}

// various common utilities

core.randi = function(min, max) {
  return Math.floor(Math.random()*(max-min)+min);
}

core.randf = function(min, max) {
  return Math.random()*(max-min)+min;
}

core.sample = function(lst) {
  var ix = core.randi(0,lst.length);
  return lst[ix];
}

// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
core.shuffle = function(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

// utilities for timing episodes
var WOB_REWARD_GLOBAL = 0; // what was reward in previous iteration?
var WOB_DONE_GLOBAL = false; // a done indicator
core.EPISODE_MAX_TIME = 10000; // in ms. Set default time to 10s.
core.EPISODE_DELAY = 500; // Time to pause before generating a new episode, in ms.

// https://stackoverflow.com/questions/3169786/clear-text-selection-with-javascript
// this piece of code clears the selection in a new episode, if a user happened
// to select some part of text. We don't want this to persist across episodes
var clearUserSelection = function() {
  if (window.getSelection) {
    if (window.getSelection().empty) {  // Chrome
      window.getSelection().empty();
    } else if (window.getSelection().removeAllRanges) {  // Firefox
      window.getSelection().removeAllRanges();
    }
  } else if (document.selection) {  // IE?
    document.selection.empty();
  }
}

core.EP_TIMER = null; // stores timer id
core.CD_TIMER = null; // stores timer ID for displaying rewards
core.ept0; // stores system time when episode begins (so we can time it)

// initialize/reset vars used for both discrete/continuous reward systems.
core.initializeVars = function(continuous){
  WOB_DONE_GLOBAL = false;
  clearUserSelection();
  if(continuous){
    core.createDisplayContinuous()
  } else {
    core.createDisplay();
  }
  core.ept0 = new Date().getTime();
  core.countdownTimer(core.EPISODE_MAX_TIME);

  if(core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); } // reset timer if needed
}

core.startEpisode = function() {
  core.initializeVars();

  // start an end of episode timer
  core.EP_TIMER = setTimeout(function(){
    core.endEpisode(-1); // time ran out
  }, core.EPISODE_MAX_TIME);
}

core.endEpisode = function(reward, time_proportional) {
  // stop timer and set to null, so that only one event gets rewarded
  // for any given episode.
  if(core.EP_TIMER !== null) {
    clearTimeout(core.EP_TIMER);
    core.EP_TIMER = null;
  } else {
    // if timer is null, don't reward anything and exit out.
    return;
  }
  var ept1 = new Date().getTime(); // get system time

  // adjust reward based on time, so acting early is encouraged
  if(typeof time_proportional === 'undefined') { time_proportional = false; }
  if(time_proportional) {
    var dt = ept1 - core.ept0; // difference in ms since start of ep
    reward = reward * Math.max(0, 1.0 - dt/core.EPISODE_MAX_TIME);
  }

  WOB_REWARD_GLOBAL += reward; // add to global, to be accessed from Python
  WOB_DONE_GLOBAL = true;
  core.logReward(reward);
  core.updateDisplay(reward);
  core.clearTimer();

  // start a new problem with a new timer. add a slight delay so that the problem
  // isn't generated immediately, which can lead to accidental clicking.
  setTimeout(function(){
    genProblem();
    core.startEpisode();
  }, core.EPISODE_DELAY);

}

core.startContinuousRewards = function() {
  core.initializeVars(true);
  // start an end of episode timer

  core.EP_TIMER = setTimeout(function(){
    core.endContinuousRewards(); // time ran out
  }, core.EPISODE_MAX_TIME);
}

core.endContinuousRewards = function() {
  // stop timer and set to null, so that only one event gets rewarded
  // for any given episode.
  if(core.EP_TIMER !== null) {
    clearTimeout(core.EP_TIMER);
    core.EP_TIMER = null;
  } else {
    // if timer is null, don't reward anything and exit out.
    return;
  }

  WOB_DONE_GLOBAL = true;
  core.clearTimer();

  // start a new problem with a new timer. add a slight delay so that the problem
  // isn't generated immediately, which can lead to accidental clicking.
  setTimeout(function(){
    genProblem();
    core.startContinuousRewards();
  }, core.EPISODE_DELAY);
}

core.continuousReward = function(reward){
  core.updateDisplayContinuous(reward);
  core.logReward(reward);
  WOB_REWARD_GLOBAL = reward; // set to reward (do not add), to be accessed from Python
}

core.logReward = function(reward){
  console.log('reward: ' + reward);
}

// returns parameters passed in the url.
// e.g. ?topic=123&name=query+string in the url would return
// QueryString["topic"];    // 123
// QueryString["name"];     // query string
// QueryString["nothere"];  // undefined (object)
core.QueryString = (function(a) {
  if (a == "") return {};
  var b = {};
  for (var i = 0; i < a.length; ++i)
  {
    var p=a[i].split('=', 2);
    if (p.length == 1)
      b[p[0]] = "";
    else
      b[p[0]] = decodeURIComponent(p[1].replace(/\+/g, " "));
  }
  return b;
})(window.location.search.substr(1).split('&'));

core.getOpt = function(d, k, def) {
  var v = d[k]
  return typeof v === 'undefined' ? def : v;
}

// template used to create the reward display HUD. This HTML
// gets wrapped inside a <div id='reward-display'> element.

core.DISPLAY_HTML = `
  <div id="reward-hide" onclick="core.hideDisplay();">X</div>
  <div class="info">
    <label>Last reward:</label>
    <span id='reward-last'>-</span>
  </div>
  <div class="info">
    <label>Last 10 average:</label>
    <span id='reward-avg'>-</span>
  </div>
  <div class="info">
    <label>Time left:</label>
    <span id='timer-countdown'>-</span>
  </div>
`;

core.DISPLAY_CONTINUOUS_HTML = `
  <div id="reward-hide" onclick="core.hideDisplay();">X</div>
  <div class="info">
    <label>Current reward:</label>
    <span id='reward-current'>-</span>
  </div>
  <div class="info">
    <label>Time left:</label>
    <span id='timer-countdown'>-</span>
  </div>
`;

// create element via JS; appending the HTML template
// directly to the body will cause jQuery UI elements
// to freak out.
core.createDisplay = function(){
  var display = document.getElementById('reward-display');
  if(display === null){
    var newDiv = document.createElement('div');
    newDiv.setAttribute('id','reward-display');
    newDiv.innerHTML = core.DISPLAY_HTML;
    document.body.appendChild(newDiv);
  }

  core.reloadDisplay();
}

core.createDisplayContinuous = function(){
  var display = document.getElementById('reward-display');
  if(display === null){
    var newDiv = document.createElement('div');
    newDiv.setAttribute('id','reward-display');
    newDiv.innerHTML = core.DISPLAY_CONTINUOUS_HTML;
    document.body.appendChild(newDiv);
  }
}

core.get_wob_scores = function(){
  var scores = localStorage.getItem('wob_scores');
  if(scores === null) return [];
  else return JSON.parse(scores);
}

core.set_wob_scores = function(wob_scores){
  var pickled = JSON.stringify(wob_scores);
  localStorage.setItem('wob_scores', pickled);
}

core.get_wob_latest = function(){
  var score = localStorage.getItem('wob_latest');
  if(score === null) return '-';
  return JSON.parse(score);
}

core.set_wob_latest = function(wob_score){
  var pickled = JSON.stringify(wob_score)
  localStorage.setItem('wob_latest', pickled);
}

// reload the display, reward stats should be persistent
// across all tasks and not just within a single task.
core.reloadDisplay = function(){
  var wob_latest = core.get_wob_latest();
  core.wob_scores = core.get_wob_scores();

  if(wob_latest !== '-'){
    var latestColor = core.computeColor(wob_latest);
    document.getElementById('reward-last').setAttribute('style', 'color: ' + latestColor);
    document.getElementById('reward-last').innerHTML = wob_latest.toFixed(2);
  }

  if(core.wob_scores.length > 0){
    var avg = core.rewardAvg();
    var avgColor = core.computeColor(avg);
    document.getElementById('reward-avg').setAttribute('style', 'color: ' + avgColor);
    document.getElementById('reward-avg').innerHTML = avg.toFixed(2);
  }
}

core.updateDisplay = function(reward){
  core.wob_scores.push(reward);
  core.wob_scores = core.wob_scores.splice(-10); // only keep the last 10 rewards.

  core.set_wob_scores(core.wob_scores);
  core.set_wob_latest(reward);

  var avg = core.rewardAvg();
  var avgColor = core.computeColor(avg);
  var latestColor = core.computeColor(reward);

  // update text and set the appropriate colors.
  document.getElementById('reward-avg').setAttribute('style', 'color: ' + avgColor);
  document.getElementById('reward-avg').innerHTML = avg.toFixed(2);
  document.getElementById('reward-last').setAttribute('style', 'color: ' + latestColor);
  document.getElementById('reward-last').innerHTML = reward.toFixed(2);
}

core.updateDisplayContinuous = function(reward){
  var latestColor = core.computeColor(reward);

  // update text and set the appropriate colors.
  document.getElementById('reward-current').setAttribute('style', 'color: ' + latestColor);
  document.getElementById('reward-current').innerHTML = reward.toFixed(2);
}

// only computes for last X tasks.
core.rewardAvg = function(){
  var toCompute = core.wob_scores.slice();
  var total = toCompute.reduce(function(a,b){ return a+b; });
  return total/toCompute.length;
}

// use RGB values for setting CSS font color.
// red value should increase as number goes towards -1
// green value should increase as number goes towards +1.
core.computeColor = function(reward){
  var red = 255;
  var green = 255;
  if(reward <= 0) green = parseInt(255*(1-Math.abs(reward)));
  else red = parseInt(255*(1-reward));
  return "rgb(" + red + "," + green + ",0);"
}

core.hideDisplay = function(){
  document.getElementById('reward-display').setAttribute('style', 'display: none');
}


core.countdownTimer = function(et){
  core.clearTimer();
  var episodeTime = et/1000;
  var currentTime = et/1000;
  var intervalTime = 1000;
  // update the timer immediately to display the total episode
  // time on start, eg. "10 / 10s"
  updateTimer();
  // set an interval so that the timer text will be updated
  // based on the `intervalTime` (ie. every 1sec)
  core.CD_TIMER = setInterval(updateTimer, intervalTime);

  function updateTimer(){
    var cd = document.getElementById('timer-countdown');
    if (currentTime <= 0){
      cd.setAttribute('style', 'color: red');
      cd.innerHTML = '0 / ' + episodeTime + 's';
      window.clearInterval(core.CD_TIMER);
      return;
    } else {
      var frac = currentTime / episodeTime;
      if(frac > 0.75) { var col = 'green'; }
      else if(frac > 0.5) { var col = 'yellow'; }
      else if(frac > 0.25) { var col = 'orange'; }
      else { var col = 'red'; }
      cd.setAttribute('style', 'color:' + col);
      cd.innerHTML = currentTime + ' / ' + episodeTime + 'sec';
      currentTime-=intervalTime/1000;
    }
  }
};

core.clearTimer = function(){
  window.clearInterval(core.CD_TIMER);
  var cd = document.getElementById('timer-countdown');
  cd.setAttribute('style', 'color: black');
  cd.innerHTML = '-';
}

// returns a score denoting how close two strings are
// to one another.
core.stringsMatch = function(expectedString, userString){

  var checkLetters = function(actualStr, userStr){
    var actualLength = actualStr.length;
    var actualSorted = actualStr.split('').sort().join();
    var userLength = userStr.length;
    var userSorted = userStr.split('').sort().join();
    var diff = Math.abs(actualLength - userLength);

    var score = 0;
    for(var i=0;i<actualLength;i++){
      score += (userSorted.indexOf(actualSorted[i]) !== -1) ?  1/actualLength : -2/actualLength;
    }

    // penalize for every additional character beyond the length
    score += userStr.length > actualLength ? (-2 * diff)/actualLength : 0;
    score = score < -1.0 ? -1.0 : score;
    return score;
  }

  var checkOrder = function(actualStr, userStr){
    var actualLength = actualStr.length;
    var userLength = userStr.length;
    var diff = Math.abs(actualLength - userLength);

    var score = 0;
    for(var i=0;i<actualLength;i++){
      score += (actualStr[i] === userStr[i]) ?  1/actualLength : -2/actualLength;
    }

    // penalize for every additional character beyond the length
    score += userStr.length > actualLength ? (-2 * diff)/actualLength : 0;
    score = score < -1.0 ? -1.0 : score;
    return score;
  }

  var checkLength = function(actualStr, userStr){
    var actualLength = actualStr.length;
    var diff = Math.abs(actualLength - userStr.length);

    var score = (actualLength - diff)/actualLength;
    score = score < -1.0 ? -1.0 : score;
    return score;
  }

  var correctLetters = checkLetters(expectedString, userString);
  var correctOrder = checkOrder(expectedString, userString);
  var correctLength = checkLength(expectedString, userString)
  var score = correctLetters*0.4 + correctOrder*0.4 + correctLength*0.2;
  return score;
}

// return a scalar based on how close the answer is to the actual answer.
core.numbersMatch = function(expectedNumber, userNumber){
  var THRESHOLD = Math.abs(0.5 * expectedNumber); // for now set this to +/- 50%.
  var diff = Math.abs(expectedNumber - userNumber);

  // special case in the event the expected number is `0`, otherwise
  // we'll get NaN errors and break everything.
  if(expectedNumber === 0 && diff === 0) return 1.0;
  else if (expectedNumber === 0 && diff < 5 ) return (5 - diff)/5;
  else if(expectedNumber === 0) return -1.0;
  else if(diff <= THRESHOLD) return (THRESHOLD - diff)/THRESHOLD;
  else return -1.0;
}
