// Package cloglik implements a Function interface that is the
// conditional log likelihood of a linear feature based model.
// The function is to be used by optimizers for finding optimal
// parameters l, mainly for text classification.
// 
// A parallel implementation of the derivative function is used
// Author: Angelos Katharopoulos <katharas@gmail.com>
package cloglik

import (
		"optimizers/gd"
		"math"
		"encoding/json"
		)

// Document contains all the information needed for training regarding
// a specific document
type Document struct {
	Features [][]int
	Class int
}

// Semantic shortcut for Features[d.Class]
func (d *Document) CorrectFeatures() []int {
	return d.Features[d.Class]
}

// Calculate the probability of class 'class' given the set of weights
// l and Document d
// P(c|d) = exp(sum(li*fi(d,c))) / sum(exp(sum(li*fi(d,c))))
func (d *Document) P(l gd.NDimensionalValue, class int) float64 {
	var (
		numerator float64
		denominator float64
		tmp float64
	)
	numerator = 0
	denominator = 0
	for cl,features := range d.Features {
		tmp = 0.0		
		for _,fi := range features {
			tmp += l[fi]
		}
		tmp = math.Exp(tmp)
		if cl==class {
			numerator += tmp
		}
		denominator += tmp
	}
	tmp = numerator/denominator
	if math.IsNaN(tmp) {
		tmp = 1
	}
	return tmp
}

// CLogLik holds all the information needed for training
// It is not to be created directly but through functions of the
// package like FromJson()
// contents:
//  - Docs       The list of documents
//  - FeatureMap A map used to translate from feature numbers back
//               to unique strings
//  - ClassMap   A map used to translate from class numbers back
//               to unique strings
//  - numerators The one time computed numerators of the derivative
//  - features_to_docs
//               A mechanism for finding all the class-document
//               combinations for which a certain feature fires
//  - Threads    How many go routines to spawn?
type CLogLik struct {
	Docs []Document
	FeatureMap []string
	ClassMap []string
	numerators gd.NDimensionalValue
	features_to_docs [][]int
	Threads int
}

// Computes numerators, is run by the functions that create CLogLik
// variables
func (p *CLogLik) computeNumerators() {
	p.numerators = make(gd.NDimensionalValue,len(p.FeatureMap))
	for _,d := range p.Docs {
		for _,fi := range d.Features[d.Class] {
			p.numerators[fi]++
		}
	}
}

// The value of the function (to satisfy the Function interface)
// Is probably going to be used for reporting progress during a training
func (p *CLogLik) Val(l gd.NDimensionalValue) float64 {
	var ret float64 = 0
	for _,doc := range p.Docs {
		ret += math.Log( doc.P(l, doc.Class) )
	}
	return ret
}

// Compute the partial derivative vector r
// For the derivative see http://nlp.stanford.edu/pubs/maxent-tutorial-slides.pdf , page 24
func (p *CLogLik) Fprime(l gd.NDimensionalValue, r gd.NDimensionalValue) {
	finish := make(chan int)
	for i,_ := range r {
		r[i] = -p.numerators[i]
	}

	classcnt := len(p.Docs[0].Features)
	doccnt := len(p.Docs)
	P_vec := make([]float64,classcnt*doccnt)
	
	docs_per_thread := int(math.Ceil(float64(len(p.Docs))/float64(p.Threads)))
	for i:=0;i<p.Threads;i++ {
		start := i*docs_per_thread
		end := int(math.Min(float64((i+1)*docs_per_thread),float64(len(p.Docs))))
		go func (start,end int, finish chan int) {
			for j:=start;j<end;j++ {
				doc := p.Docs[j]
				for class,_ := range doc.Features {
					P_vec[class*doccnt + j] = doc.P(l,class)
				}
			}
			finish <- 1
		}(start,end,finish)
	}
	
	fcount := 0
	for fcount<p.Threads {
		select {
			case <- finish:
				fcount++
		}
	}
	
	features_per_thread := int(math.Ceil(float64(len(l))/float64(p.Threads)))
	for i:=0;i<p.Threads;i++ {
		start := i*features_per_thread
		end := int(math.Min(float64((i+1)*features_per_thread),float64(len(l))))
		go func (start,end int, finish chan int) {
			for j:=start;j<end;j++ {
				for _,doc := range p.features_to_docs[j] {
					r[j] += P_vec[doc]
				}
			}
			finish <- 1
		}(start,end,finish)
	}
	
	fcount = 0
	for fcount<p.Threads {
		select {
			case <- finish:
				fcount++
		}
	}
}

// Helper function, very useful to create valid function points
func (p *CLogLik) DimCnt() int {
	return len(p.numerators)
}

// Create a CLogLik instance from a Json object that contains all the
// necessary data for training.
//
// It is a lot easier to create tokenizers, feature functions etc in
// dynamic string oriented languages. In addition, training is a not so
// frequent phenomenon, one mostly creates a model for use not a
// dataset for training.
//
// As a result, it is useful to create nlp packages in any language and
// use Go for training only. One data format could be Json.
// The following format is parsed here (each feature is, as it is
// common, represented by a unique string):
//
// [
//    {
// 	    "class1": ["feature1","feature2",...,"featuren"],
// 	    "class2": ["feature1","feature2",...,"featuren"],
// 	    .
// 	    .
// 	    "classn": ["feature1","feature2",...,"featuren"],
// 	    "__label__": "classk"
//    },
//    {
// 	    "class1": ["feature1","feature2",...,"featuren"],
// 	    "class2": ["feature1","feature2",...,"featuren"],
// 	    .
// 	    .
// 	    "classn": ["feature1","feature2",...,"featuren"],
// 	    "__label__": "classk"
//    }
// ]
func FromJson (raw []byte) *CLogLik {
	var p CLogLik
	
	var tmpDocs []map[string]interface{}
	err := json.Unmarshal(raw,&tmpDocs)
	if err != nil || len(tmpDocs)==0 {
		return nil
	}
	
	classes := make(map[string]int)
	features := make(map[string]int)
	p.Docs = make([]Document,len(tmpDocs))
	for k,_ := range tmpDocs[0] {
		if k != "__label__" {
			classes[k] = len(classes)
		}
	}
	for di,d := range tmpDocs {
		df := make([][]int,len(classes))
		for k,v := range d {
			if k == "__label__" {
				p.Docs[di].Class = classes[v.(string)]
			} else {
				class := classes[k]
				df[class] = make([]int, len(v.([]interface{})));
				for i,fi := range v.([]interface{}) {
					if _,ok := features[fi.(string)]; !ok {
						features[fi.(string)] = len(features)
					}
					df[class][i] = features[fi.(string)]
				}
			}
		}
		p.Docs[di].Features = df
	}
	p.features_to_docs = make([][]int,len(features))
	for i,_ := range p.features_to_docs {
		p.features_to_docs[i] = make([]int,0)
	}
	doccnt := len(p.Docs)
	for i,doc := range p.Docs {
		for class,features := range doc.Features {
			for _,fi := range features {
				p.features_to_docs[fi] = append(p.features_to_docs[fi],class*doccnt+i)
			}
		}
	}
	p.FeatureMap = make([]string,len(features))
	p.ClassMap = make([]string,len(classes))
	for class,int_repr := range classes {
		p.ClassMap[int_repr] = class
	}
	for feature,int_repr := range features {
		p.FeatureMap[int_repr] = feature
	}
	
	p.computeNumerators()
	p.Threads = 1
	
	return &p
}


