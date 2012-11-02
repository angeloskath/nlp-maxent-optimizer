package main

import (
		"optimizers/gd"
		"nlp/cloglik"
		"io/ioutil"
		"os"
		"runtime"
		"encoding/json"
		"fmt"
		)

func main() {
	
	var raw []byte
	if len(os.Args)<3 {
		raw,_ = ioutil.ReadAll(os.Stdin)
	} else {
		raw,_ = ioutil.ReadFile(os.Args[1])
	}
	
	// read in the training data
	maxent := cloglik.FromJson(raw)
	
	// initialize gradient descent parameters
	grad_descent := gd.GradientDescent{}
	grad_descent.LearningRate = 0.1
	grad_descent.Threshold = 0.01
	grad_descent.MaxIter = -1
	
	// parallelism
	//threads,_ := strconv.ParseInt(os.Args[3],0,0)
	//fmt.Println("Threads: ",threads)
	threads := runtime.NumCPU()
	runtime.GOMAXPROCS(threads)
	grad_descent.Threads = threads
	maxent.Threads = threads
	
	// starting point
	x0 := make(gd.NDimensionalValue,maxent.DimCnt()) // x0 = [0,...,0]
	
	// minimize
	xmin := grad_descent.Optimize(maxent,x0)

	fmt.Fprintln(os.Stderr,"Loglikelihood: ",maxent.Val(xmin))

	// prepare the result for JSON output
	rs := make(map[string]float64)
	for i,v := range xmin {
		rs[maxent.FeatureMap[i]] = v
	}
	
	// output
	json_out, _ := json.Marshal(rs)
	if len(os.Args)<3 {
		os.Stdout.Write(json_out)
	} else {
		ioutil.WriteFile(os.Args[2],json_out,0666)
	}
}
