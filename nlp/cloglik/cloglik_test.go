package cloglik

import (
		"testing"
		"optimizers/gd"
		"math"
		//"fmt"
		)

func TestMaxent(t *testing.T) {
	
	const json = `
	[
		{
			"A": ["a"],
			"B": ["b"],
			"__label__": "A"
		},
		{
			"A": ["b"],
			"B": ["a"],
			"__label__": "B"
		}
	]
	`
	c := FromJson([]byte(json))
	
	opt := gd.GradientDescent{1,0.01,-1,1}
	
	start := make(gd.NDimensionalValue,c.DimCnt())
	weights := opt.Optimize(c,start)

	if math.Abs( weights[0] + weights[1] ) > 0.01 {
		t.Errorf("%v=>%v\n%v=>%v",c.FeatureMap[0],weights[0],c.FeatureMap[1],weights[1])
	}
}


