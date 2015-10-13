__author__ = 'auroua'
import sys
import matlab
import matlab.engine

eng = matlab.engine.start_matlab()

def basic_test(eng):
  print "Basic Testing Begin"
  print "eng.power(100,2) = %d"%eng.power(100,2)
  print "eng.max(100,200) = %d"%eng.max(100,200)
  print "eng.rand(5,5) = "
  print eng.rand(5,5)
  print "eng.randi(matlab.double([1,100]),matlab.double([3,4]))"%\
    eng.randi(matlab.double([1,100]),matlab.double([3,4]))
  print "Basic Testing End"

def plot_test(eng):
  print "Plot Testing Begin"
  eng.workspace['data'] =  \
    eng.randi(matlab.double([1,100]),matlab.double([30,2]))
  eng.eval("plot(data(:,1),'ro-')")
  eng.hold('on',nargout=0)
  eng.eval("plot(data(:,2),'bx--')")
  print "Plot testing end"


if __name__=='__main__':
    # basic_test(eng)
    I=eng.imread('rice.png')
    mapping=eng.getmapping(8,'riu2')

    lbpMap = eng.lbp(I, 2, 8, 0, 0)

    H1=eng.lbp(I,1,8,mapping,'h')
    print H1
    eng.subplot(2,1,1)
    eng.stem(H1)
    H2=eng.lbp(I)
    eng.subplot(2,1,2)
    eng.stem(H2)
    # a = matlab.double([1,4,9,16,25],[3,4,5,6,7,8])
    SP=matlab.int32([[-1,-1],[-1,0],[-1,1],[0,-1],[-0,1],[1,-1],[1,0],[1,1]])
    I2=eng.lbp(I,SP,0,'i')

    eng.quit()