from oslo_model import OsloModel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as opt
from scipy.stats import norm
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import time as tm
from scipy.stats import linregress

def task_1():    
    
    #define parameters
    num_grains = int(input('enter number of grains => '))
    seed = int(input('enter seed for rng => '))
    probs = input('enter probabilities => ')
    probs = [float(prob) for prob in probs.split()]
    if max(probs) > 1.0 or min(probs) < 0.0:
        raise Exception('probabilities must be between 0.0 and 1.0')
    lengths = input('enter lengths => ')
    lengths = [int(length) for length in lengths.split()]
    
    
    critical_times = []
    i = 0
    avg_heights_total = np.empty((3, lengths[0]))
    colors = ['#9fb8ad', '#383e56', '#fb743e']
    
    #collect data to demonstrate code is working as expected
    
    for prob in probs:
        plt.title('total heights vs time')
        plt.xlabel('time')
        plt.ylabel('total height')
        for length in lengths:
            if length < 8:
                continue
            if length % 8 != 0:
                continue
            sim = OsloModel(length, prob, seed)
            tc, heights,_ = sim.run(num_grains)
            avg_heights = sim.get_average_heights()

            print('probability = ', prob)
            print('length =', length, 'height at site 1', avg_heights[0])
            print('critical time', tc)
            critical_times.append(tc)
            avg_heights_total[i,:] = avg_heights[:lengths[0]]
        i += 1
        
    #plot visualisations for systems
     
    x = np.arange(0, lengths[0], 1)
    for i in range(np.size(probs)):
        if i==1:
            plt.bar(x, avg_heights_total[i], align='edge', label=r'$P = %0.1f$'%(probs[i]), color=colors[i], width=0.8)
        else:
            plt.bar(x, avg_heights_total[i], align='edge', label=r'$P = %0.1f$'%(probs[i]), color=colors[i], width=0.8)
    plt.xlim(-0.5, lengths[0]+0.5)
    plt.title('Steady State System Visualisations for L = 64', fontsize = 19)
    plt.legend(fancybox='True', fontsize = 'x-large')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(r'$h_i$', fontsize = 18)
    plt.xlabel(r'$i$', fontsize = 18)
    plt.show()
    
    
def task_2ABD():
    
    #initialise parameters
    num_grains = int(input('enter number of grains => '))
    seeds = input('enter seeds for rng => ')
    seeds = [int(seed) for seed in seeds.split()]
    prob = float(input('enter probability => '))
    lengths = input('enter lengths => ')
    lengths = [int(length) for length in lengths.split()]
    
    times = np.linspace(1, num_grains, num_grains, endpoint=True)      
    numpyarray = np.empty((np.size(lengths), num_grains))
    critical_times_mean = np.empty(np.size(lengths))

    i=0
    #collect data
    for length in enumerate(lengths):
        critical_times = []
        avg_heights = [0] * num_grains
        for seed in seeds:
            sim = OsloModel(length[1], prob, seed)
            tc, heights, _ = sim.run(num_grains)
            avg_heights = [avg_heights[i] + heights[i] for i in range(num_grains)]
            critical_times.append(tc)
        num_seeds = len(seeds)
        avg_heights = [float(avg_heights[i]) / num_seeds for i in range(num_grains)]
        numpyarray[length[0],:] = avg_heights
        critical_times_mean[i]= np.sum(critical_times)/ num_seeds 
        i += 1
      
    #2A
       
    # plot system height as a function of time with the average heights shown as horizontal lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    fig1, ax1=plt.subplots()    
    for i in range(np.shape(numpyarray)[0]):
          plt.plot(times[:100000], numpyarray[i,:100000], label=r'$L= %i$'%(lengths[i]))
    
    horiz = critical_times
    
    for j in range(np.shape(numpyarray)[0]):
        plt.hlines(horiz[j],  -4998.950000000001, 100000, color = colors[j], linestyles='dotted')
    
    plt.xlabel('t', fontsize =20)
    plt.ylabel(r'$\tilde{h}(t;L)$', fontsize = 20)
    ax1.set_yscale('log')
    ax1.set_yticks([1, 10, 100])
    plt.ylim(1, 598.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 'x-large', loc='lower right')
    plt.title(' System Height as a Function of Time', fontsize = 19)
    plt.savefig('System Height as a Function of Time.png')
    plt.show()
    
    #2B

    # find the critical times scaling
    plt.clf()
    
    plt.plot(lengths, np.sqrt(critical_times_mean), markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5)
    plt.xlabel('L', fontsize = 20)
    plt.ylabel(r'$\sqrt{\langle t_c(L) \rangle}$', fontsize = 20)
    plt.title(r'Square Root of Cross-Over Time as a function of L', fontsize = 19)

    model = sm.OLS(np.sqrt(critical_times_mean), lengths)
    results = model.fit()
    
    plt.plot(lengths, results.params * np.array(lengths), color = '#fb743e',linewidth=1.7, linestyle = '-', label = r'$\sqrt{\langle t_c(L) \rangle}$ = 0.9329 L' )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 'x-large', loc='lower right')
    plt.grid(color='gray', linestyle='-.', linewidth=0.7)
    plt.savefig('Square Root of Cross-Over Time as a function of L.png')
    plt.show()
    
    #2D
    
    # Data collapse of system height
    
    #have to set parameters as saved big data
        
    x = np.load('Data Collapse array.npy')
    numpyarray = x
    lengths = np.array([4, 8, 16, 32, 64, 128, 256])
    plt.clf()
    
    plt.set_cmap('hot')
    transformedarray = np.empty((7, 250000))
    transformedtimes = np.empty((7, 250000))
    times = times = np.linspace(1, 250000, 250000, endpoint=True)      
    
    
    fig, ax=plt.subplots()
    
    for j in range(np.shape(numpyarray)[0]):
        transformedarray[j,:] = numpyarray[j,:] / lengths[j]
        transformedtimes[j,:] = times / (lengths[j])**2
        ax.loglog(transformedtimes[j,:], transformedarray[j,:], label=r'$L= %i$'%(lengths[j]))
   
    xaxis = np.arange(8.7e-6, 0.86, 0.0001)     
    locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
    ax.xaxis.set_major_locator(locmaj)
    locminx = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2, 0.4,0.6,0.8),numticks=12)
    ax.xaxis.set_minor_locator(locminx)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 'x-large', loc='lower right')     
    plt.ylabel(r'$\frac{\tilde{h}(t;L)}{L}$', fontsize = 20)
    plt.xlabel(r'$\frac{T}{L^2}$', fontsize = 20)
    plt.title('Data Collapse for System Height', fontsize = 19)
    plt.show()
    
 
    #fig1, ax1=plt.subplots()
    
    transformedtimessmall = []
    for t in transformedtimes[6,:]:
        if t < 0.85 and t > 0.0017:
            transformedtimessmall.append(t)
        else:
            continue
        
    np.shape(transformedtimessmall)
    transformedarraysmall = transformedarray[6,111:55705]
    #ax1.loglog(transformedtimessmall, transformedarraysmall)
    
    logx = np.log(transformedtimessmall)
    logy = np.log(transformedarraysmall)
    coeffs, pcov = np.polyfit(logx,logy,deg=1, cov=True)
    poly = np.poly1d(coeffs)
    yfit = lambda x: np.exp(poly(np.log(x)))
    ax.loglog(xaxis,yfit(xaxis), color = 'black',linewidth=1.7, linestyle = 'dashed', label = r'$ Slope = 0.5092 \pm 0.00001$')
    plt.show()
    plt.savefig('Data Collapse for System Height.png')

def task_2EFG():
        num_grains = int(input('enter number of grains => '))
        seed = int(input('enter seed for rng => '))
        prob = float(input('enter probability => '))

        lengths = np.array([4, 8, 16, 32, 64, 128, 256])

        avg_heights = []
        std_devs = []
        hists = []
        heights = []
        for length in lengths:
            sim = OsloModel(length, prob, seed)
            tc, heights_, _ = sim.run(num_grains)
            n = num_grains - tc
            if n == 0:
                continue
            avg_height = np.mean(heights_[tc+1:])
            avg_heights.append(avg_height)
            sd = np.std(heights_[tc+1:])
            std_devs.append(sd)
            print('length = ', length, 'crossover time = ', tc, 'avg height', avg_height, 'std dev = ', sd)
            hists.append(np.histogram(heights_[tc:], density=True))
            heights.append(heights_[tc:])
            

        def fit_func_avg(x, a1, a2, w1) -> np.array:
            return a1 * x * (1.0 - a2 * np.power(x, -w1))

        def fit_func_sd(x, a, b) -> np.array:
            return a * np.power(x, b)

        
#2E 

# estimate a0 from L>>1 scaling        

        plt.clf()     
                
        plt.plot(lengths, avg_heights, markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5 )
        coeffs, cov = np.polyfit(lengths[4:], avg_heights[4:], deg=1, cov=True)

        line = np.poly1d(coeffs)
        plt.ylabel(r'$\langle h(t;L) \rangle$', fontsize = 20)
        plt.xlabel('L', fontsize = 20)
        plt.title('Average System Height as a Function of L', fontsize=19)
        xaxis=np.arange(0,260,1)
        plt.plot(xaxis, line(xaxis), color = '#fb743e',linewidth=1.7, linestyle = '-', label = r'$ Slope = 1.726 \pm 0.0009$')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize = 'x-large', loc='lower right')
        plt.grid(color='gray', linestyle='-.', linewidth=0.7)
        plt.show()
        plt.savefig('Average Height as a Function of L.png')
        
# estimate a0 from polyfit 

# fit the avg heights to the formula
        popt, pcov = opt.curve_fit(fit_func_avg, lengths, avg_heights)
        fitted_vals = fit_func_avg(np.array(lengths), popt[0], popt[1], popt[2])

        
#estimate a1 and w1 using updated a0

        plt.clf()
               
        numpyarray = np.empty((np.size(lengths), 1))
        numpyarray2 = np.empty((np.size(lengths), 1))
        plt.clf()
        for length in enumerate(lengths):
             numpyarray[length[0],:] = avg_heights[length[0]]/(coeffs[0]*length[1])
             numpyarray2[length[0],:] = avg_heights[length[0]]/(1.7340128606510277*length[1])
             
           
        yaxis= np.log(1 - numpyarray)
        yaxis2= np.log(1 - numpyarray2)
        plt.loglog(lengths, np.e**yaxis, markersize = 8, marker='X', color = 'grey', linestyle = 'None', zorder=2.5, label= r'$a_0 = 1.726 $')
        plt.loglog(lengths, np.e**yaxis2, markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5, label = r'$a_0 = 1.734 $' )
        
        xaxis=np.arange(3.8, 280, 1)
        coeffs_try2, cov_try= np.polyfit(np.log(lengths), yaxis2, deg=1, cov=True)
           
        coeffs_22 = np.array([float(coeffs_try2[0]), float(coeffs_try2[1])])
        line_22 = np.poly1d(coeffs_22)
        poly = line_22
        yfit = lambda x: np.exp(poly(np.log(x)))
        
        plt.loglog(xaxis,yfit(xaxis), color = '#fb743e',linewidth=1.7, linestyle = '-', label = r'$Slope = -0.5888 \pm 0.0009 $')
        
        plt.legend(fontsize = 'x-large', loc='lower left')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(color='gray', linestyle='-.', linewidth=0.7)
        plt.xlabel('L', fontsize = 20)
        plt.ylabel(r'$1- \frac{\langle h(t;L) \rangle}{a_0L}$', fontsize=20)
        plt.title('Plot to estimate Corrections to Scaling Parameters', fontsize =19)
        
        
        coeffs_try = np.polyfit(np.log(lengths), yaxis, deg=1)
        coeffs_2 = np.array([float(coeffs_try[0]), float(coeffs_try[1])])   
        line_2 = np.poly1d(coeffs_2)
        plt.show()
        plt.savefig('Plot for A1 and W1 from polyfit A0 value.png')
        print(np.poly1d(line_2), 'is the equation')
        print('which means that a1 is', np.e**coeffs_2[1], 'and w1 is', -coeffs_2[0])
        print('and that a0 is ', coeffs[0])
        print('or these answers for better fit:', np.e**coeffs_22[1], -coeffs_22[0], 1.7340128606510277 )

# 2F

        # fit the std devs to a power law aL^b
        plt.clf()
        popt, pcov = opt.curve_fit(fit_func_sd, lengths, std_devs)
        fitted_vals = fit_func_sd(np.array(lengths), popt[0], popt[1])
        
        # plot to demonstrate
 
        plt.xlabel('L', fontsize = 20)
        plt.ylabel(r'$\sigma_h(L)$', fontsize = 20)
        plt.loglog(lengths, std_devs, '+', label = r'$\sigma_h(L) = \sqrt{\langle h^2(t;L) \rangle - \langle h(t;L) \rangle ^2}$', markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5)
        yint = np.arange(0, 3, 1)
        plt.yticks(yint)
        plt.loglog(lengths, fitted_vals, label = r'$\sigma_h(L) = {%.3f}L^{%.3f} $' % (float("0.5840"), float("0.2393")), color = '#fb743e',linewidth=1.7, linestyle = '-')
        plt.legend(fontsize = 'x-large', loc='lower right')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(color='gray', linestyle='-.', linewidth=0.7)
        plt.title('Standard Deviation Scaling for L', fontsize =19)
        plt.show()
        plt.savefig('Standard Deviation vs Length.png')
        
        
# 2G 

        #plot the heights as a function of system size to demonstrate CLT
        plt.clf()
        i=0   
        mus = []
        sigmas = []
        bin_no = 14
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        xdata = np.empty((np.size(lengths), bin_no))
        ydata = np.empty((np.size(lengths), bin_no))
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('h', fontsize = 20)
        plt.ylabel('P(h;L)', fontsize=20)
        plt.title('Probability Distribution for System Height', fontsize =19)
        for heights_ in heights:
            
            counts, bins= np.histogram(heights_, density=True, bins=bin_no)
            centers = 0.5*(bins[1:]+ bins[:-1])
            mu, sigma = norm.fit(heights_)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, bin_no)
            plt.plot(x, norm.pdf(x, mu, sigma), label=r'$L= %i$'%(lengths[i]))
            plt.vlines(avg_heights[i], min(norm.pdf(x, mu, sigma)), max(norm.pdf(x, mu, sigma)), color=colors[i], linestyles='dotted', linewidth=1)
            plt.xlim((0, 460))
            plt.ylim((0, 0.6))
            plt.show()
            ydata[i,:] = counts
            xdata[i,:] = centers
            mus.append(mu)
            sigmas.append(sigma)
            i+=1
        plt.legend(fontsize = 'x-large', loc='upper right')
        plt.grid(color='gray', linestyle='-.', linewidth=0.7)    
        plt.show()
        plt.savefig('Probability Distribution for System Height.png')
        
        
        # data collapse for 2g
        plt.clf()
        i=0
        mus=[]
        for heights_ in heights:
                mu, sigma = np.mean(heights_), np.std(heights_)
                heights_ = np.array(heights_)
                heights_ = (heights_ - mu) / sigma
                kde = gaussian_kde(heights_)
                x = np.unique(heights_)
                plt.plot(x, kde(x)*sigma/(np.sqrt(2*np.pi)*0.57), zorder=-1, label=r'$L= %i$'%(lengths[i]))
                
                i+=1      
                mus.append(mu)
                
        plt.show()
        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color='black', linewidth=1.2, zorder=2, label = 'Normal Distribution')
        plt.legend(fontsize = 'x-large', loc='upper right')
        plt.grid(color='gray', linestyle='-.', linewidth=0.7)   
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.xlabel(r'$\frac{h-\langle h(t;L) \rangle}{\sigma_h}$', fontsize=20)
        plt.ylabel(r'$P(h;L)\sigma_h$', fontsize = 20)
        plt.title('Data Collapse for Probability of System Height', fontsize=19)
        plt.show()      
        plt.savefig('data collapse for height prob.png')

#ACF                 
        #autocorrelation function calculation
        num_grains = 100000
        seed = 123456
        prob = 0.5
        lengths = [4, 8, 16, 32, 64, 128, 256]
        
        avg_heights = []
        std_devs = []
        heights = []
        for length in lengths:
            sim = OsloModel(length, prob, seed)
            tc, heights_, _ = sim.run(num_grains)
            if tc == num_grains:
                continue
            avg_height = np.mean(heights_[tc + 1:])
            avg_heights.append(avg_height)
            sd = np.std(heights_[tc + 1:])
            std_devs.append(sd)
            heights.append(heights_[tc:])
            z = np.array(sim.local_slopes)
            acf = sm.tsa.acf(z, nlags=3)
            print('acf of slopes = ', acf)
            z = np.random.uniform(size=10)
            acf = sm.tsa.acf(z, nlags=3)
            print('acf of uniform rand = ', acf)    


def data_binning(avalanches, a_fac: float = 1.1) -> ():
    max_size = np.amax(avalanches)
    n = avalanches.size
    num_bins = int(np.log10(max_size) / np.log10(a_fac)) + 1
    edges = np.zeros((num_bins+1))
    edges[0] = 0.0
    for i in range(1, num_bins+1):
        edges[i] = a_fac**i

    counts, edges = np.histogram(avalanches, edges)
    sizes = np.array([np.sqrt(edges[i] * edges[i+1]) for i in range(num_bins)])
    sizes[0] = np.sqrt(a_fac)
    probs = np.zeros(counts.size,dtype='float')
    for i in range(num_bins):
        delta_s = edges[i+1] - edges[i] + 1
        probs[i] = float(counts[i]) / float(n * delta_s)
    return probs, sizes


def task_3a():
    num_grains = 500000
    seed = 12349
    prob = 0.5
    lengths = [4, 8, 16, 32, 64, 128, 256]

    plt.xlabel('avalanche size')
    plt.ylabel('prob')

    f = open('avalanches.txt', 'w')
    f_raw = open('avalanches_raw.txt', 'w')
    critical_sizes = []

    for length in lengths:
        tic = tm.perf_counter()
        sim = OsloModel(length, prob, seed)
        tc, _, avalanches = sim.run(num_grains)
        avalanches = np.array(avalanches[tc:])
        f_raw.write('length = '+str(length)+' tc = '+str(tc)+'\n')
        for avalanche in avalanches:
            f_raw.write(str(avalanche)+' ')
        f_raw.write('\n')
        avalanches = avalanches[avalanches > 0.0]
        avalanche_probs, avalanche_sizes = data_binning(avalanches, 1.25)
        idx = np.nonzero(avalanche_probs)
        avalanche_probs = avalanche_probs[idx]
        avalanche_sizes = avalanche_sizes[idx]
        avalanche_probs = avalanche_probs / np.sum(avalanche_probs)
        crit = avalanche_sizes.size - 3
        critical_sizes.append(avalanche_sizes[crit])
        f.write('length = ' + str(length) + '\n')
        for size, p in zip(avalanche_sizes, avalanche_probs):
            f.write(str(size) + ' ' + str(p) + '\n')
        plt.plot(np.log10(avalanche_sizes), np.log10(avalanche_probs))
        toc = tm.perf_counter()
        print('length = ', length, ' num secs to run = ', str(toc-tic))

    f_raw.close()
    f.close()
    plt.show()
    plt.savefig('log prob vs log size.png')
    d_, intercept, r_val, p_val, std_err = linregress(np.log10(lengths), np.log10(critical_sizes))
    print('slope', d_,'intercept', intercept)

def task_3a_analysis():
    length = 0
    data = []
    with open('avalanches_raw.txt', 'r') as f:
        for line in f:
            tokens = line.split()
            if tokens[0] != 'length':
                tokens = np.array([float(size) for size in tokens])
                data.append((length, tokens))
            else:
                length = int(tokens[2])
    f.close()
    plt.clf()
    fig, ax1 = plt.subplots(1)
    ax1.set_xlabel('s', fontsize=20)
    ax1.set_ylabel(r'$\tilde{P}(s;L)$', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Avalanche Size Probability Distribution', fontsize=19)
    critical_sizes = []
    lengths = []
    log_data = {}
    for d in data:
        length = d[0]
        avalanches = np.array(d[1])
        avalanches = avalanches[avalanches > 0.0]
        avalanche_probs, avalanche_sizes = data_binning(avalanches, 1.25)
        idx = np.nonzero(avalanche_probs)
        avalanche_probs = avalanche_probs[idx]
        avalanche_sizes = avalanche_sizes[idx]
        avalanche_sizes = np.log10(avalanche_sizes)
        avalanche_probs = np.log10(avalanche_probs)
        crit = avalanche_sizes.size - 3
        critical_sizes.append(avalanche_sizes[crit])
        lengths.append(length)
        ax1.loglog(np.power(10,avalanche_sizes), np.power(10,avalanche_probs), label=r'$L= %i$'%(length))
        log_data[length] = (avalanche_sizes, avalanche_probs)
    plt.legend(fontsize = 'x-large', loc='upper right')
    d_, intercept, r_val, p_val, std_err = linregress(np.log10(lengths), critical_sizes)
    print('d from critical sizes is = ', d_,'intercept', intercept)

    plt.clf()
    fig3, ax3 = plt.subplots(1)
    ax3.set_xlabel('L', fontsize=20)
    ax3.set_ylabel(r'$s_c$', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Critical Size Scaling', fontsize=19)
    ax3.loglog(lengths, lengths**d_, label = r'$s_c = L^{%.3f} $' % (float(d_)), color = '#fb743e',linewidth=1.7, linestyle = '-')
    ax3.loglog(lengths, 10**np.array(critical_sizes), markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5)
    plt.legend(fontsize = 'x-large', loc='lower right')
    plt.grid(color='gray', linestyle='-.', linewidth=0.7)
    plt.show()
    plt.savefig('Critical Size Scaling.png')
    
    tau, intercept = 0, 0
    sizes = log_data[256][0] - d_ * np.log10(256.0)  # scale with L^d
    probs = log_data[256][1]
    sizes = sizes[10:43]
    probs = probs[10:43]
    t, intercept, r_val, p_val, std_err = linregress(-sizes, probs)
    tau = t
    print('tau =', tau, 'this is the slope of 256', 'error is', std_err)
    
    
    tau = tau
    d_ = d_
    plt.clf()
    fig2, ax2 = plt.subplots(1)
    for length in log_data.keys():
        s_crit = length ** d_
        sizes = log_data[length][0]
        probs = log_data[length][1]
        probs = probs + tau * sizes
        sizes = sizes - np.log10(s_crit)
        ax2.loglog(np.power(10,sizes), np.power(10,probs), label=r'$L= %i$'%(length))
    ax2.set_xlabel(r'$\frac{s}{L^D}$', fontsize=20)
    ax2.set_ylabel(r'$\tilde{P}(s;L) s^{\tau_s}$', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Data Collapse for Avalanche Size Probability', fontsize=19)
    plt.legend(fontsize = 'x-large', loc='lower left')
    plt.show()
    plt.savefig('Data Collapse for Avalanche Size Prob.png')


def task_3b():
    
    # moment estimates
    # read data from file
    length = 0
    data = []
    with open('avalanches_raw.txt', 'r') as f:
        for line in f:
            tokens = line.split()
            if tokens[0] != 'length':
                tokens = np.array([float(size) for size in tokens])
                data.append((length, tokens))
            else:
                length = int(tokens[2])
    f.close()

    num_moments = 4
    moments_data = {}
    for d in data:
        length = d[0]
        avalanches = np.array(d[1])
        avalanches = avalanches[avalanches > 0.0]
        sizes, counts = np.unique(avalanches, return_counts=True)
        # calculate moments
        moments = np.zeros(num_moments, dtype='float')
        for k in range(num_moments):
            s_k = np.power(sizes, k+1) # k starts at 0 so need to add 1
            moments[k] = s_k @ counts
            moments[k] /= np.sum(counts)
        moments_data[length] = moments

    plt.clf()
    
    #create plots for moment scaling analysis next to eachother
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel(r'$L$', fontsize=20)
    ax1.set_ylabel(r'$\langle s_k \rangle$', fontsize=20)
    ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=16)
    ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=16)
    ax1.set_title('Kth Moment Analysis' , fontsize=19)
    ax1.grid(color='gray', linestyle='-.', linewidth=0.7)
    color=['#6d0404','#9d3a2a','#cd6851','#ff9386']
    slopes = []
    for i in range(num_moments):
        x, y = [], []
        for length in moments_data.keys():
            y.append(moments_data[length][i])
            x.append(length)
        x = np.log10(x)
        y = np.log10(y)
        # regression of moments vs length
        ax1.loglog(10**np.array(x), 10**np.array(y), label=r'$k= %i$'%(i+1), marker='o', color=(color[i]), linestyle='None')
        
        slope, intercept, r_val, p_val, std_err = linregress(x[3:], y[3:])
        slopes.append(slope)
        xaxis=np.arange(3, 260,1)
        ax1.loglog(xaxis, 10**(np.array(np.log10(xaxis))*slope+intercept), color=(color[i]) )
        ax1.legend(fontsize = 'x-large', loc='upper left')
        print('slope', slope, intercept)



    ax2.set_xlabel(r'$k$', fontsize=20)
    ax2.set_ylabel(r'$D(1+k-\tau_s)$',fontsize=20)
    x = np.linspace(1, num_moments, num_moments, endpoint=True)
    y = np.array(slopes)
    ax2.plot(x, y, markersize = 10, marker='X', color = '#383e56', linestyle = 'None', zorder=2.5)
    # regression of slopes vs k
    slope, intercept, r_val, p_val, std_err = linregress(x, y)
    d = slope
    tau = 1.0 - intercept/slope
    print('tau =', tau, 'd =', d, 'this is for 3b')
    ax2.plot(x, slope*x + intercept,  color = '#fb743e',linewidth=1.7, linestyle = '-', label = r'$D(1+k-\tau_s) = {%.3f}k {%.3f} $' % (float(slope), float(intercept)))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(r'$D(1+k-\tau_s) vs. k$', fontsize=19)
    plt.legend(fontsize = 'x-large', loc='lower right')
    plt.grid(color='gray', linestyle='-.', linewidth=0.7)
    plt.show()
    plt.savefig('Moment Scaling Analysis.png')
    


if __name__ == '__main__':
  
    #task_1()
    #task_2ABD()
    #task_2EFG()
    #task_3a()
    #task_3a_analysis()
    #task_3b()
    
       