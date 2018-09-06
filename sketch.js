//x^n-1 in polys
let n = 5;
//arr to hold vars for func
let vars = [];
//how fine the resolution is
let segments = 100;
//points
let xList = [];
let yList = [];

let anneal = false;


let learningRate = 1;
let curRate = learningRate;
//adam SGD algo
let optimizer = tf.train.adam(learningRate);

let step = 0;

//arrow notation loss function
const loss = (pred, label) => pred.sub(label).square().mean();

//enumesque thing in js
const r = Object.freeze({
    POLY:   Symbol("poly"),
    EXP:    Symbol("exp"),
    TRIG:   Symbol("cos"),
    INV:    Symbol("inv"),
});

//init current regression to polynomial
let curReg = r.POLY;

function setup() {
    let canvas = createCanvas(500,500);
    canvas.parent('holder');
    canvas.mousePressed(addPoint);
    resetVars();
}

function draw() {
    background(50);

    //draw the points
    stroke(255);
    strokeWeight(10);
    for(let i = 0; i < xList.length; i++){
        let x = map(xList[i], -1, 1, 0, width);
        let y = map(yList[i], -1, 1, height, 0);
        point(x, y);
    }
    
    strokeWeight(1);
    line(width / 2, 0, width / 2, height);
    line(0, height / 2, width, height / 2);
    
    if(xList.length > 0) {
        let yTensor = tf.tensor1d(yList);
        //minimize func 10 times a frame. parameterize?
        for(let i = 0; i < 10; i++){
            optimizer.minimize(() => loss(predict(xList), yTensor));
        }
    }    
    
    //init x points arr for the func
    let arr = [];
    for(let i = 1; i <= segments; i++){
        arr.push(map(i, 1, segments, -1, 1));
    }
    
    //make a tensor by predicting the x arrays y values
    const yTensor = tf.tidy(() => predict(arr));
    //synchronize & send to array
    let yCurve = yTensor.dataSync();
    //clean up
    yTensor.dispose();
    //draw the func
    beginShape();
    noFill();
    strokeWeight(2);
    for(let i = 0; i < arr.length; i++){
        let x = map(arr[i], -1, 1, 0, width);
        let y = map(yCurve[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();
    //check if equation changed
    let equation = getEquation();
    let e = document.getElementById("equation");
    e.innerHTML = equation;
    
    //check if we're "annealing"(is that the right word?) and if so begin to slow the training rate
    step++;
    if(step % 100 == 0 && anneal){
        curRate /= 5;
        optimizer = tf.train.adam(curRate);
    }
}

function mouseDragged() {
    
}
function mousePressed() {
}

function addPoint(){
    xList.push(map(mouseX, 0, width, -1, 1));
    yList.push(map(mouseY, 0, height, 1, -1));
    curRate = learningRate;
    anneal = false;
}

function predict(xs) {
    const xTensor = tf.tensor1d(xs);
    //y = ((ax + b)x + c)x + d
    //y = (ax + b)x + c
    //y = ax + b
    //y = a;
    let yTensor;
    
    /* Exponential */
    /* y = a * b^x */
    if(curReg == r.EXP)
        yTensor = tf.tidy(() => vars[0].mul(vars[1].pow(xTensor)));
    
    //Suprisingly, the first one i did was the hardest
    //making it scalable to any size polynomial was not as trivial as i thought it would be.
    //mostly just edge cases lol
    /* Polynomial */
    else if (curReg == r.POLY){
        if(n == 1) yTensor = tf.tidy(() => xTensor.mul(tf.scalar(0)).add(vars[0]));
        else {
            yTensor = tf.tidy(() => {
                
                    let tmp = xTensor.mul(vars[0]).add(vars[1]);
                    for(let i = 1; i < vars.length - 1; i++)
                        tmp = xTensor.mul(tmp).add(vars[i + 1]);
                return tmp;
            });
        }
    }
    /* trig */
    /* y = a*cos(b*x+c) */
    else if (curReg == r.TRIG){
        yTensor = tf.tidy(() => {
           return vars[0].mul(vars[1].mul(xTensor).add(vars[2]).cos()); 
        });
    }
    /* inverse */
    /* y = a / x */
    else if (curReg == r.INV){
        yTensor = tf.tidy(() => {
            return vars[0].div(xTensor);
        });
    }
    return yTensor;
}

//There has to be a better way
function changeRegression(){
    let e = document.getElementById("regressionList");
    let reg = e.options[e.selectedIndex].text;
    console.log("setting regression to " + reg);
    if(reg == "Exponential"){
        n = 2;
        vars.length = 0;
        vars.push(tf.variable(tf.scalar(random(1))));
        vars.push(tf.variable(tf.scalar(random(1))));
        curReg = r.EXP;
    } else if (reg == "Polynomial"){
        curReg = r.POLY;
    } else if (reg == "Trig"){
        curReg = r.TRIG;
        n = 3;
    } else if (reg == "Inverse"){
        curReg = r.INV;
        n = 1;
    }
    resetVars();
}

function resetVars(){
    vars.length = 0;
    for(let i = 0; i < n; i++){
        vars.push(tf.variable(tf.scalar(random(-1, 1))));
    }
    optimizer = tf.train.adam(learningRate);
    anneal = false;
}

function changeN(){
    let e = document.getElementById("nTextBox");
    let newN = e.value;
    if(curReg != r.EXP && newN >= 0){
        n = newN;
        n++;
        resetVars();
    }
}

function clearPoints(){
    xList.length = 0;
    yList.length = 0;
    resetVars();
}

//should break this out into other methds
function getEquation(){
    let result = "";
    if(curReg == r.POLY){
        let i = 0;
        while(i < n - 1){
            if(i != 0) result += " + ";
            let val = vars[i].dataSync();
            result += val +  " * x ^ " + (n - 1 - i);
            i++;
        }
        let val = vars[i].dataSync();
        if(i != 0) result += " + ";
        result +=  val;
    } else if(curReg == r.EXP){
        let a = vars[0].dataSync();
        let b = vars[1].dataSync();
        result = a + " * " + b + " ^ x"
    } else if(curReg == r.TRIG){
        let a = vars[0].dataSync();
        let b = vars[1].dataSync();
        let c = vars[2].dataSync();
        result = a + " * cos(" + b + " * x) + " + c;
    } else if(curReg == r.INV){
        let a = vars[0].dataSync();
        result = a + " / x";
    }
    return result;
}

function fineGrain(){
    anneal = true;
}

function changeRate(){
    let e = document.getElementById("rateBox");
    let newRate = e.value;
    if(newRate >= 0){
        learningRate = newRate;
        optimizer = tf.train.adam(learningRate);
    }
}