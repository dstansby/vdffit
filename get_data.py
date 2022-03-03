from sunpy.net import Fido
from sunpy.net import attrs as a

result = Fido.search(a.Time('2019-01-01', '2022-01-01') & a.soar.Product('MAG-RTN-NORMAL'))
print(result)

Fido.fetch(result, path='/Volumes/Work/Data/solo/mag')
