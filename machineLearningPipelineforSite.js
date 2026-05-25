//naturally put classes at the top

class MLMetrics{ //evaluate accuracy

    static Report(actual,predicted){
        let tp = 0, tn = 0, fp = 0, fn = 0;
        /*
        where:
         t is true, 
         f is false, 
         p is positive, 
         n is negative
         This rates the model's success- shows how it failed (confusion matrix), accuracy
        */
        
        for (let i=0;i<actual.length;i++){

            const p = predicted[i];const a = actual[i] 
            if (p === 1 && a === 1){ //correct
                tp++
            }
            else if (p === 0 && a === 0){
                tn++
            } 
            else if (p === 1 && a === 0){//incorrect
                fp++
            }
            else if (p === 0 && a === 1){
                fn++
            }; 
        }; //values of confusion matrics, which shows an accuracy report. 

        const total = actual.length;
        
        //Formulas
        const accuracy = (tp + tn) / total;
        const precision = tp / (tp + fp) || 0; //In case of divide by 0
        const recall = tp / (tp + fn) || 0; //successful 'catches' (finding fraud)
        const f1 = 2 * (precision * recall) / (precision + recall) || 0; 

        return{ //this returns the scores that evaluate the model's success

            confusionMatrix: {tp, tn, fp, fn}, 
            accuracy: accuracy.toFixed(4), 
            precision: precision.toFixed(4), 
            recall: recall.toFixed(4), 
            f1Score: f1.toFixed(4) 
        };
    };
};

//--------------------------------------------------------------------------------------------------------------------------


/*
No need to understand the math at the macro level, only the logical processes in coding terms. 
Since we're using what is called logistic regression analysis to classify (0 or 1, True or False) our data, we use complex, university-level maths. 
Instead, I can take logic from existing programs to mold into my learner to be able to use. 
This class' objects, when created, will allow me to use its functions to essentially create a 'brain' (or neural network)
*/


class Learner{ 
    constructor(cCount){ //making an array of each 'column' from the dataset

        let ar = [];
        for (let i=0;i<cCount;i++){
            ar.push(0) 
        };
        this.weights = ar; 
        this.baseline = 0.1; 
        

    };

    _getPercentage(n){ //this is basically crafting the sigmoid function - this takes index of list 
    //this is a cool function that I found online, which almost emulate the beginnings of a sigmoid function.

        let sigmoid = 1 / (1 + Math.exp(-n)); 
        return sigmoid; 
    };

    _prediction(features){

        let sum = 0
        for (let i=0;i<this.weights.length;i++){
            sum += this.weights[i] * (features[i] || 0);
        };
        sum += this.baseline; 
        return this._getPercentage(sum);
    }

    learn(data,real){ //this is also the idea behind gradient descent
        //this is the main function of the learner class.
        
        let alpha = 0.0007; //learning rate optimized
        let iterations = 10; //times machine loops through

        const numRows = data.length;
        

        for (let i=0;i<iterations;i++){

            for (let j=0;j<numRows;j++){

                const rowFeatures = data[j];
                const actual = real[j];
                const prediction = this._prediction(rowFeatures);

                const error = actual-prediction;

                this.baseline += alpha*error;
                for (let k=0;k<rowFeatures;k++){ //crucial to update every feature's weight per row. Remember this is a 2-d array

                    let check = alpha * rowFeatures[k]; 
                    //because I take weights for each feature of the 'column' (or transactions training)
                    
                    
                    this.weights[k] += check
                    
                };
            };
        };
    };
};

//recognize that Case will be composed in the pattern 'learner' model later, so theres no need for any inheritance from learner model.
class Case{ //where Case is the 'flow controller,' serving as each individual 'leaf' in the tree. 

    constructor(predicate,action){ //condition and result
        this.predicate = predicate; 
        this.action = [action];
        this.nomatch = {match: false,value: NaN}; //map to classify transaction
    };

    eval(object){
        
        
        if (typeof this.predicate === 'function'){ 
            isMatch = this.predicate(object);
        } 
        else{
            isMatch = this.predicate;
        }
        
        if (isMatch){ //just for better readability
            return this.handle(object);
        }
        else{
            return this.nomatch; 
        }
    };

    handle(){ //hoisted in eval. 
        
        if (Array.isArray(this.action)){ 
            //we're working with arrays to build a 'recursive decision tree,' where we can say not only say 'if x then y', but if 'w,x,z then y.'
            
            for (let i=0;i<length(this.action);i++){ 
                //rule allows me to run through each rule in an array. This works because each 'sub-rule' in the array is also this object. 
                
                let l = eval(i)
                if (l.match){
                    return l; //return l such that it evaluates as match when run through the above function
                };   
            };
            return this.nomatch //if no match
        };

        return{
            match: true, 
            value: this.action
        } //classification if there is no array 
    }; 
};







function split(data){ //this split data is a filter for our code.

    

    for (let i=data.length-1;i>0;i--){ //probably not the best way to do this, but it makes most sense to me. 

        let j = Math.floor(Math.random()*(i+1)); //also probably not the best way
        [data[i],data[j]] = [data[j],data[i]]; //swap i/j 
    };

    let a = data.length/10;
    a = Math.floor(a);
    let test = data.slice(1*a);
    let train = data.slice(0,9*a);

    return{
        train,test
    };
}



/*
Understanding our data is crucial. In the data we use, less than .001% of the data is fraudulent. 
This means that in the metrics objects that were made earlier, a model would be right 99.999% of the time just by guessing 'not fraudulent.'
We need to know the actual percentages, which is impossible to do manually after I've scrambled the data. This is my code to work with the class imbalance.
*/

//I'll use this function on data before both training and testing. 
function scaler(data){ //this is a 'robust scaler.' I can maximize the precision I work with this way.
    
    let temp = data.toSorted(
        (a,b) => a - b
    ); //sort least to greatest

    let l = data.length;

    let q2 =  temp[Math.floor(l * 0.5)]; //median
    let iqr = (temp[Math.floor(l * 0.75)]) - (temp[Math.floor(l * 0.25)]); //interquantile range, which I've worked with in the past. 

    if (iqr == 0){
            iqr++;
        } ;//let iqr = 1 to prevent dividing by 0
    
    let nD = [] //new data
    for (let i=0;i<data.length;i++){

        if (typeof data[i] !== 'string'){
            nD[i] = (data[i]-q2)/iqr
        }
        else{
            nD[i] = data[i]
        }
    };
    return nD;
}

//this will allow me to know which columns from the dataset are worth working with- I dont want to obfuscate my decision tree with irrelevant objects
function RetrieveWeights(q,c){  
    
    let useful = []; //to be returned
    for (let i =0;i<q.weights.length;i++){

        let features ={
            cname: c[i], //take column name
            strength: Math.abs(q.weights[i]),  //since weight could be confused with weights
            isRelavent: q.weights[i] > 0 //bool 
        };
        
        useful.push(features); 
        
    };

    useful.sort((a,b) => b.strength - a.strength); //sort for each column by priority (weight)
    return useful;
}





//-----------------------------------------------------------------------------------------------------------



// lets work with the papaparse library first, which lets us work with the csv data we're using. 

import Papa from 'papaparse';

//webstreams api

Papa.parse("creditcard.csv",{ 
        //understand that papaparse is an async fnction, so it works with the data as the code runs- if we dont do this,  brain will 'think' without seeing the data.
        header: true, 
        dynamicTyping: true, //papaparse advertises this feature- it basically executes python's eval function and turns strings into numbers
        download: true, //this change is required without a filestream.
        complete: function(csvData){ //this is our complete function, which is actually where all logic goes (we're only working with one file.)
            console.log("begun data...")
            const {train,test} = split(csvData.data);
        
            const lKeys = csvData.meta.fields 
            const cNames = lKeys.filter(i => i !== 'Class' && i !== 'Time'); //for now I want to work with the data, not class. 
            

            const tAmounts = test.map(row => row.Amount);
            test.forEach((row, i) => row.Amount = scaler(tAmounts[i]));

            const trainNames = train.map(row => cNames.map(n => row[n])); //names for row
            const trainClass = train.map(row => row.Class); //class for row

            const testClass = test.map(row => row.Class);
            const brain = new Learner(cNames.length);
            console.log("learning...")
            
            //loop through matrices like so
 
            brain.learn(trainNames,trainClass); 

  

        /*
        Using my class "Case", I need to create different rules and label them. 
        I can do this with this formatting: 
        >> const rule = (predicate,action) => new Case(predicate,action)

        By looping nodes into the action, I can create a 'tree'-like code, hence the name 'decision tree.' This ressembles this:
        const rule3 = (predicate,action) => new Case(predicate,[rule1,rule2])
        Here, the array is the next set of 'checks' down the line. 
        The nested objects are be confusing, but it is simpler than dynamically creating functions in the decision tree.
        */
        


        const highRisk = new Case(row => brain._prediction(cNames.map(n => row[n])) > 0.05, 'BLOCK'); //with action being block, and predicate checking each feature
        const lowRisk = new Case(row => brain._prediction(cNames.map(n => row[n])) <= 0.05, 'ALLOW');

        
        const decisionTree = new Case(
            row => true, //such that row is true
            row =>{ //check
                const h = highRisk.eval(row);
                if (h.value === 'BLOCK'){
                    return 'BLOCK'};
                if (h.value === "AllOW"){
                    const l = lowRisk.eval(row)
                    return l.value
                }
                
            }
        );
     //array of possible cases. This is a root node, and will be the 'gate' for the sub-nodes
        

        const predictions = test.map(row =>{
            const result = decisionTree.eval(row);
        if (result.value === 'BLOCK'){
            return 1
        }
        else{
            return 0
        }
        });
        

        const stats = MLMetrics.Report(testClass,predictions); //using mlmetrics like a function
        const importanceReport = RetrieveWeights(brain,cNames); 
        console.log('Primary contributing features (column):');
        console.table(importanceReport,stats); //show top 5
        

    }

}); 


//turn object into readable text
