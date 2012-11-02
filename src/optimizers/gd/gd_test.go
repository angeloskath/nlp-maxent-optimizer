package gd

import (
		"testing"
		"math"
		)

type Parabola struct {
	a,b,c float64
}
func (p *Parabola) Fprime(x NDimensionalValue, r NDimensionalValue) {
	r[0] = 2*p.a*x[0] + p.b
}
func (p *Parabola) Val(x NDimensionalValue) float64 {
	return p.a*x[0]*x[0] + p.b*x[0] + p.c
}

type NDimensionalParabola struct {
	Parabolas []Parabola
}
func (ndp *NDimensionalParabola) Fprime(x NDimensionalValue, r NDimensionalValue) {
	for i,p := range ndp.Parabolas {
		p.Fprime(x[i:i+1],r[i:i+1])
	}
}
func (ndp *NDimensionalParabola) Val(x NDimensionalValue) float64 {
	v := 0.0
	for i,p := range ndp.Parabolas {
		v += p.Val(x[i:i+1])
	}
	return v
}

func TestSingleDimension(t *testing.T) {
	const precision = 0.001
	
	p := &Parabola{5,6,3}
	gd := GradientDescent{0.01,precision,100,1}
	r := gd.Optimize(p,make(NDimensionalValue,1))
	if math.Abs(r[0]+0.6) > precision {
		t.Errorf("min(5x^2+6x+3)=%v should be within %v of 0.6",r[0],precision)
	}
}

func Test3Dimensions(t *testing.T) {
	const precision = 0.001
	
	p1 := Parabola{5,6,3}
	p2 := Parabola{1,-6,4}
	p3 := Parabola{5,2,-34} 
	
	ps := [...]Parabola{p1,p2,p3}
	p := &NDimensionalParabola{ps[:]}
	
	gd := GradientDescent{0.01,precision,1000,1}
	r := gd.Optimize(p,make(NDimensionalValue,3))
	
	if math.Abs(r[0]+0.6) > precision {
		t.Errorf("min(5x^2+6x+3)=%v should be within %v of -0.6",r[0],precision)
	}
	if math.Abs(r[1]-3) > precision {
		t.Errorf("min(x^2-6x+4)=%v should be within %v of 3",r[1],precision)
	}
	if math.Abs(r[2]+0.2) > precision {
		t.Errorf("min(5x^2+2x-34)=%v should be within %v of -0.2",r[2],precision)
	}
}


