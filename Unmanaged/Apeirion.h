#ifndef WORLDS5_APEIRON_H
#define WORLDS5_APEIRON_H
/*
 *	Apeirion Engine:  5-dimensional routines
 *
 *	Andrew G Williams  April 1996
 */

#define v_mov(a,b)														\
	b[0] = a[0];                                                        \
	b[1] = a[1];                                                        \
	b[2] = a[2];                                                        \
	b[3] = a[3];                                                        \
	b[4] = a[4]

#define v_mul(a,b)																		\
	double b0 = b[0];																			\
	b[0] = a[0]*b0              - a[1]*(b[1]-b[2]+b[3]-b[4]) + b[1]*(a[2]-a[3]+a[4]);	\
	b[1] = a[0]*b[1] + a[1]* b0 + a[2]*(b[2]-b[3]+b[4])      - b[2]*(a[3]-a[4]);		\
	b[2] = a[0]*b[2] + a[2]* b0 - a[3]*(b[3]-b[4])           + b[3]* a[4];				\
	b[3] = a[0]*b[3] + a[3]* b0 + a[4]* b[4];											\
	b[4] = a[0]*b[4] + a[4]* b0

#define v_mulm(a,b,c)                                     								 \
	c[0] = a[0]*b[0]              - a[1]*(b[1]-b[2]+ b[3]-b[4]) + b[1]*(a[2]-a[3]+a[4]); \
	c[1] = a[0]*b[1] + a[1]* b[0] + a[2]*(b[2]-b[3]+b[4])       - b[2]*(a[3]-a[4]);		 \
	c[2] = a[0]*b[2] + a[2]* b[0] - a[3]*(b[3]-b[4])            + b[3]* a[4];			 \
	c[3] = a[0]*b[3] + a[3]* b[0] + a[4]* b[4];											 \
	c[4] = a[0]*b[4] + a[4]* b[0]

// Multiply vector a by a constant
#define v_mulc(a,b)																		\
	a[0] = a[0]*b;														\
	a[1] = a[0]*b;														\
	a[2] = a[0]*b;														\
	a[3] = a[0]*b;														\
	a[4] = a[0]*b

#define v_add(a,b)                                     					\
	b[0] = a[0] + b[0];                                                 \
	b[1] = a[1] + b[1];                                                 \
	b[2] = a[2] + b[2];                                                 \
	b[3] = a[3] + b[3];                                                 \
	b[4] = a[4] + b[4]

#define v_addm(a,b,c)                                   				\
	c[0] = a[0] + b[0];                                                 \
	c[1] = a[1] + b[1];                                                 \
	c[2] = a[2] + b[2];                                                 \
	c[3] = a[3] + b[3];                                                 \
	c[4] = a[4] + b[4]

#define v_subm(a,b,c)                                   				\
	c[0] = b[0] - a[0];                                                 \
	c[1] = b[1] - a[1];                                                 \
	c[2] = b[2] - a[2];                                                 \
	c[3] = b[3] - a[3];                                                 \
	c[4] = b[4] - a[4]

#define v_clr(a)														\
	a[0] = 0;                                                         	\
	a[1] = 0;                                                       	\
	a[2] = 0;                                                         	\
	a[3] = 0;                                                         	\
	a[4] = 0                                                          	\

#define v_mod(a)                                 		               	\
	(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]+a[3]*a[3]+a[4]*a[4])

#define v_mandel(a,b)   \
	if(DimTotal>5)		\
		v_mandelx(a,b);	\
	else				\
	{					\
		double a0 = a[0];		\
		a[0] =   a0*a0   - a[1]*(a[1]-a[2]+a[3]-a[4]) + a[1]*(a[2]-a[3]+a[4]) + b[0]; \
		a[1] = 2*a0*a[1] + a[2]*(a[2]-a[3]+a[4])	  - a[2]*(a[3]-a[4])      + b[1]; \
		a[2] = 2*a0*a[2] - a[3]*(a[3]-a[4])			  + a[3]*(a[4])           + b[2]; \
		a[3] = 2*a0*a[3] + a[4]*(a[4])										  + b[3]; \
		a[4] = 2*a0*a[4]													  + b[4]; \
	}

void v_movx(double *a, double *b)
{
	extern int DimTotal;

  for(int i = 0; i < DimTotal; i++)
	{	
	  b[i] = a[i];
  }
}

void v_mulx(double *a, double *b)
{
	extern int DimTotal;
	double sum1,sum2;
	double products;
	double a0 = a[0];

	for(int x=0; x<DimTotal; x++)
	{	
		sum1=0; sum2=0;

		for(int i=x+1; i<DimTotal; i+=2)
		{	
			sum1 = sum1 + b[i] - b[i+1];
			sum2 = sum2 + a[i+1] - a[i+2];
		}

		if(double(x)/2 == x/2)
		{	
			sum1 = -sum1;
			sum2 = -sum2;
		}
		
		products = a0 * b[x];
		if(x>0) products = products + a[x] * b[0];

		a[x] = products + a[x+1] * sum1 - b[x+1] * sum2;
	}
}

void v_mandelx(double *a, double *b)
{
	extern int DimTotal;
	double sum1,sum2;
	double products;
	double a0 = a[0];

	for(int x=0; x<DimTotal; x++)
	{	
		sum1=0; sum2=0;

		for(int i=x+1; i<DimTotal; i+=2)
		{	
			sum1 = sum1 + a[i] - a[i+1];
			sum2 = sum2 + a[i+1] - a[i+2];
		}

		if(float(x)/2 == x/2)
		{	
			sum1 = -sum1;
			sum2 = -sum2;
		}

		products = a0 * a[x];
		if(x>0) products *= 2;

		a[x] = products + a[x+1] * sum1 - a[x+1] * sum2 + b[x];
	}
}

void v_invm(double *a, double *b)
{
	double gra, grb, grc, grd, x;

	if (a[0]==0 && a[2]==0 && a[3]==0 && a[4]==0 && a[1]!=0)
	{
		b[0]=b[2]=b[3]=b[4]=0;
		b[1]=-1/a[1];
	}
	else
	{
		if (a[0]==0)
			b[0]=b[1]=b[2]=b[3]=b[4]=1e16;
		else
		{
			gra=-a[4];
			grb=-a[3]-gra*a[4]/a[0];
			grc=(grb*(a[3]-a[4])-gra*a[3])/a[0]-a[2];
			grd=(grb*a[2]-gra*a[2]-grc*(a[2]-a[3]+a[4]))/a[0]-a[1];
			x=a[0]*a[0]-grd*(a[1]-a[2]+a[3]-a[4])+a[1]*(grc-grb+gra);

			if (x==0)
				b[0]=b[1]=b[2]=b[3]=b[4]=1e16;
			else
			{
				if (x>1e20) x=1e18;
				if (x<1e-20) x=1e-18;
				x=1/x;
				b[0]=x*a[0];
				b[1]=x*grd;
				b[2]=x*grc;
				b[3]=x*grb;
				b[4]=x*gra;
			}
		}
	}
}
#endif

