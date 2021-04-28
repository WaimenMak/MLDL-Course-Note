class Nodes{
   public:
      Nodes * last_right;
      Nodes * last_left;
      Nodes * next;
      void *value;
      Nodes * grad;
      Nodes * sub_grad;
      bool require_grad;
      bool need_update;
      bool BN;
      
  
      Nodes(void){
         this.last_right=NULL;
         this.last_left=NULL;
         this.next=NULL;
         this.value=NULL;
         this.grad=NULL;
         this.sub_grad=NULL;
         this.require_grad=true;
         this.need_update=false;
         this.BN=false;
      }
      ~Nodes(void){}
};

class Input: public Nodes{
   public:
      Input(double& X[][]){
         int y = 1;
         //this.value = y; 
      }
      
      ~Input(void){}
};
