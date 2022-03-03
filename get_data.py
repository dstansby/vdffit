from sunpy.net import Fido, attrs as a
import sunpy_soar

result = Fido.search(a.Time('2019-01-01', '2022-01-01') & a.soar.Product('MAG-RTN-NORMAL'))
print(result)

Fido.fetch(result, path='/Volumes/Work/Data/solo/mag')
