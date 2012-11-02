// Package gd implements gradient descent optimization method
package gd

import "math"

// NDimensionalValue is just an alias for an array of floats
type NDimensionalValue []float64

// Function is an interface that contains one function that computes
// the partial derivatives of a function
type Function interface {
	Val(v NDimensionalValue) float64
	Fprime(v NDimensionalValue, r NDimensionalValue)
}

// GradientDescent holds all the configuration for the GD
type GradientDescent struct {
	LearningRate, Threshold float64
	MaxIter, Threads int
}

// gd compute routine receives messages of this type
type message struct {
	start,end int
}
// computeRoutine is a function to be run as a goroutine
// it computes the new point given a point and a vector of partial
// derivatives (x,fp)
// TODO Maybe start,end should be given at the beginning and not by
//      messaging
func (gd GradientDescent) computeRoutine (m chan message, computation_finished, terminate chan int, optimized *bool, fp,x NDimensionalValue) {
	done := false
	var msg message
	for !done {
		select {
			case <-terminate:
				done = true
				continue
			case msg = <-m:
				// nothing go on
		}

		for i:=msg.start;i<msg.end;i++ {
			x[i] -= gd.LearningRate*fp[i]
			*optimized = *optimized && (math.Abs(fp[i]) < gd.Threshold)
		}

		computation_finished <- 1
	}
}

// Optimize implements batch gradient descent.
// The simplest optimization algorithm ever.
func (gd GradientDescent) Optimize(f Function, x NDimensionalValue) NDimensionalValue {
	iter := 0
	optimized := false
	dim := len(x)
	fp := make(NDimensionalValue, dim)
	items_per_thread := int(math.Ceil( float64(dim)/float64(gd.Threads) ))
	finish := make(chan int, gd.Threads)
	terminate := make([](chan int), gd.Threads)
	msg := make([](chan message), gd.Threads)
	for i,_ := range terminate {
		terminate[i] = make(chan int)
		msg[i] = make(chan message)
		go gd.computeRoutine(msg[i], finish, terminate[i], &optimized, fp, x)
	}
	for !optimized && gd.MaxIter != iter {
		optimized = true
		
		f.Fprime(x, fp)
		
		for i:=0;i<gd.Threads;i++ {
			msg[i] <- message{i*items_per_thread,int(math.Min(float64((i+1)*items_per_thread),float64(dim)))}
		}

		// wait for the routines to finish
		for i:=0;i<gd.Threads;i++ {
			<-finish
		}
		
		// this has been replaced by the goroutines above
		//for i,fpi := range fp {
		//	x[i] -= gd.LearningRate*fpi
		//	optimized = optimized && (math.Abs(fpi) < gd.Threshold)
		//}
		
		iter += 1
	}
	
	// terminate the goroutines
	for _,t := range terminate {
		t<-1
	}
	return x
}

