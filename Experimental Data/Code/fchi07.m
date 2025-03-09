function [chi chimg ci P Q]=fchi07(X,mn,ri,r,mask,plotit)

if(~exist('plotit','var')||isempty(plotit))
  plotit=false;
end

A=X(1);
K=X(2);
P=cos(0);  %X(3)
Q=sin(0);  %X(3)
x=36;
y=0;

ci=(A*sin(K*peipf2(r,x,y,P,Q)).^2+mn);
ci(isnan(ci))=mn;
chimg=(ri-ci).*mask;
chi=mean(abs(chimg(:)));

if(plotit)
  %display(chi)
  simage([ri.*mask ci.*mask abs(chimg)]);
  fprintf('%f\n',chi);
  drawnow;
end
