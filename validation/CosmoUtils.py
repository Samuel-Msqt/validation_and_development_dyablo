
import numpy as np

################################################################ 

def field2Pk(field,LboxMpc,nsamp=64):
    
    # Returns the power spectrum of a 3D field
    # LboxMpc is the comiving size of the box
    # nsamp is the number of frequencies for the power spectrum sampling

    N=np.shape(field)[0] # assumes 2D cubic field
    k1=np.fft.fftfreq(N,d=1/N)*2*np.pi/LboxMpc
    kx,ky,kz=np.meshgrid(k1,k1,k1,indexing='ij')
    k=np.sqrt(kx**2+ky**2+kz**2)

    deltak=np.fft.fftn(field)/N**3
    d2=np.real(deltak*np.conj(deltak))

    kmin=np.min(np.abs(k1))
    kmax=np.max(np.abs(k1))
    print(kmin,kmax)
    #bk=np.linspace(kmin,kmax,num=nsamp)
    bk=np.logspace(np.log10(1e-3),np.log10(kmax),num=nsamp)
    bkcen=0.5*(bk[1:]+bk[:-1])
    H1,bb1=np.histogram(k,bins=bk)
    H2,bb2=np.histogram(k,bins=bk,weights=d2)
    Pk=H2/((H1==0)+H1)*LboxMpc**3

    return Pk,bkcen
################################################################

def field2Pk_CIC(field,LboxMpc,nsamp=64,Npart=128**3):

    # Returns the power spectrum of a 3D field
    # This version includes a correction for Poisson and CIC window (eq. 2.10 arXiv:2403.13561v1)
    # LboxMpc is the comiving size of the box
    # nsamp is the number of frequencies for the power spectrum sampling

    N=np.shape(field)[0] # assumes 2D cubic field
    k1=np.fft.fftfreq(N,d=1/N)*2*np.pi/LboxMpc
    kx,ky,kz=np.meshgrid(k1,k1,k1,indexing='ij')
    k=np.sqrt(kx**2+ky**2+kz**2)

    deltak=np.fft.fftn(field)/N**3
    d2=np.real(deltak*np.conj(deltak))
    H=LboxMpc/N
    kN=(np.pi/H) #nyquist freq
    C1=(1.0-2.0/3.0*np.sin(np.pi*kx/(2*kN))**2)*(1.0-2.0/3.0*np.sin(np.pi*ky/(2*kN))**2)*(1.0-2.0/3.0*np.sin(np.pi*kz/(2*kN))**2)
    d2=d2/C1


    kmin=np.min(np.abs(k1))
    kmax=np.max(np.abs(k1)) #kN
    print(kmin,kmax)
    #bk=np.linspace(kmin,kmax,num=nsamp)
    bk=np.logspace(np.log10(1e-3),np.log10(kmax),num=nsamp)

    bkcen=0.5*(bk[1:]+bk[:-1])
    H1,bb1=np.histogram(k,bins=bk)
    H2,bb2=np.histogram(k,bins=bk,weights=d2)
    H3,bb3=np.histogram(k,bins=bk,weights=d2**2)

    Pk=(H2/((H1==0)+H1))*LboxMpc**3
    EPk=np.sqrt((H3/((H1==0)+H1))-(H2/((H1==0)+H1))**2)*LboxMpc**3/np.sqrt(((H1==0)+H1)) #sigma/sqrt(N)

    return Pk,bkcen,EPk

################################################################
def part2cic(positions,NC,contrast=True):

    #CIC density estimate over a NC*NC*NC grid


    b=np.linspace(0,NC,NC+1)

    dxrho=1./NC
    idx=np.int32(positions/dxrho)
    rhodm=np.zeros((NC,NC,NC))

    d=positions/dxrho-idx
    t=1.0-d

    dt,edens=np.histogramdd((idx+np.array((0,0,0)))%NC,bins=(b,b,b),weights=t[:,0]*t[:,1]*t[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((1,0,0)))%NC,bins=(b,b,b),weights=d[:,0]*t[:,1]*t[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((0,1,0)))%NC,bins=(b,b,b),weights=t[:,0]*d[:,1]*t[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((0,0,1)))%NC,bins=(b,b,b),weights=t[:,0]*t[:,1]*d[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((1,1,0)))%NC,bins=(b,b,b),weights=d[:,0]*d[:,1]*t[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((0,1,1)))%NC,bins=(b,b,b),weights=t[:,0]*d[:,1]*d[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((1,0,1)))%NC,bins=(b,b,b),weights=d[:,0]*t[:,1]*d[:,2])
    rhodm+=dt
    dt,edens=np.histogramdd((idx+np.array((1,1,1)))%NC,bins=(b,b,b),weights=d[:,0]*d[:,1]*d[:,2])
    rhodm+=dt

    if(contrast):
        deltadm=(rhodm-np.average(rhodm))/np.average(rhodm)
        print("avg rho", np.average(rhodm))
        print(np.sum(rhodm))
        return deltadm, rhodm
    else:
        return rhodm

########################################################################
def readgrafic(fname):
    
    #grafic files reader


    f=open(fname,'rb')
    dt=np.dtype([
        ('dummy1','i4'),
        ('n',('i4',3)),
        ('dx','f4'),
        ('offset',('f4',3)),
        ('aexp','f4'),
        ('om','f4'),
        ('ov','f4'),
        ('H0','f4'),
        ('dummy2','i4'),
        ('dummy3','i4'),

        ])
    header=np.fromfile(f,dtype=dt,count=1)

    N=header['n'][0][0]
    delta=np.zeros((N,N,N),dtype=np.float32)

    dslice=np.dtype([
        ('d1','i4'),
        ('s',('f4',N*N)),
        ('d2','i4'),
    ])
    f.seek(44+8)
    rawdata=np.fromfile(f,dtype=dslice,count=N)

    for p in range(N):
        delta[:,:,p]=np.reshape(rawdata[p]['s'],(N,N))
    aexp=header['aexp'][0]
    om=header['om'][0]
    ov=header['ov'][0]
    H0=header['H0'][0]
    dx=header['dx'][0]
    h=header['H0'][0]/100.0
    Lbox=N*dx

    return delta,Lbox,aexp,om,ov,H0
