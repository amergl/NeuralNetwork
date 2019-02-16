def assertEquals(expected,actual,delta):
    difference=abs(expected-actual)
    assertValue=difference < delta
    if not assertValue:
        print("expected %e but was %e"%(expected,actual))
    assert assertValue